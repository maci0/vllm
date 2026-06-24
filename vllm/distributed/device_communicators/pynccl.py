# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


# ===================== import region =====================
import threading

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

import vllm.envs as envs
from vllm.distributed.device_communicators.pynccl_wrapper import (
    NCCLLibrary,
    buffer_type,
    cudaStream_t,
    ncclComm_t,
    ncclDataTypeEnum,
    ncclRedOpTypeEnum,
    ncclUniqueId,
)
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger
from vllm.utils.torch_utils import current_stream

logger = init_logger(__name__)

_NCCL_SYMM_OPS_REGISTERED = False


def register_nccl_symmetric_ops(pynccl_comm):
    from vllm.utils.torch_utils import direct_register_custom_op

    global _NCCL_SYMM_OPS_REGISTERED
    if _NCCL_SYMM_OPS_REGISTERED:
        return
    _NCCL_SYMM_OPS_REGISTERED = True

    def all_reduce_symmetric_with_copy_impl(input_tensor: torch.Tensor) -> torch.Tensor:
        # Persistent registered scratch (sized in warmup) so the captured hot
        # path never allocates/registers a new NCCL window -> CUDA-graph safe.
        symm_input = pynccl_comm.get_symm_scratch(
            "ar_in", input_tensor.numel(), input_tensor.dtype, input_tensor.device
        ).view(input_tensor.shape)
        symm_output = pynccl_comm.get_symm_scratch(
            "ar_out", input_tensor.numel(), input_tensor.dtype, input_tensor.device
        ).view(input_tensor.shape)
        symm_input.copy_(input_tensor)
        pynccl_comm.all_reduce(symm_input, symm_output)
        # Copy out of the reused scratch into a normally-managed tensor.
        return symm_output.clone()

    def all_reduce_symmetric_with_copy_fake(input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(input_tensor)

    direct_register_custom_op(
        op_name="all_reduce_symmetric_with_copy",
        op_func=all_reduce_symmetric_with_copy_impl,
        fake_impl=all_reduce_symmetric_with_copy_fake,
    )

    # Symmetric-memory (NVLS multicast) reduce_scatter / all_gather along dim 0.
    # Routes NCCL to the ncclSymk LDMC/STMC kernels instead of generic RING.
    def reduce_scatter_symmetric_with_copy_impl(
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        ws = pynccl_comm.world_size
        chunk = input_tensor.shape[0] // ws
        out_shape = (chunk, *input_tensor.shape[1:])
        symm_input = pynccl_comm.get_symm_scratch(
            "rs_in", input_tensor.numel(), input_tensor.dtype, input_tensor.device
        ).view(input_tensor.shape)
        symm_output = pynccl_comm.get_symm_scratch(
            "rs_out",
            input_tensor.numel() // ws,
            input_tensor.dtype,
            input_tensor.device,
        ).view(out_shape)
        symm_input.copy_(input_tensor)
        pynccl_comm.reduce_scatter(symm_output, symm_input)
        # Copy out of the symmetric scratch so the result is a normally-managed
        # tensor (the scratch is reused by the next call).
        return symm_output.clone()

    def reduce_scatter_symmetric_with_copy_fake(
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        chunk = input_tensor.shape[0] // pynccl_comm.world_size
        return torch.empty(
            (chunk, *input_tensor.shape[1:]),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )

    direct_register_custom_op(
        op_name="reduce_scatter_symmetric_with_copy",
        op_func=reduce_scatter_symmetric_with_copy_impl,
        fake_impl=reduce_scatter_symmetric_with_copy_fake,
    )

    def all_gather_symmetric_with_copy_impl(
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        ws = pynccl_comm.world_size
        out_shape = (input_tensor.shape[0] * ws, *input_tensor.shape[1:])
        symm_input = pynccl_comm.get_symm_scratch(
            "ag_in", input_tensor.numel(), input_tensor.dtype, input_tensor.device
        ).view(input_tensor.shape)
        symm_output = pynccl_comm.get_symm_scratch(
            "ag_out",
            input_tensor.numel() * ws,
            input_tensor.dtype,
            input_tensor.device,
        ).view(out_shape)
        symm_input.copy_(input_tensor)
        pynccl_comm.all_gather(symm_output, symm_input)
        return symm_output.clone()

    def all_gather_symmetric_with_copy_fake(
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        return torch.empty(
            (input_tensor.shape[0] * pynccl_comm.world_size, *input_tensor.shape[1:]),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )

    direct_register_custom_op(
        op_name="all_gather_symmetric_with_copy",
        op_func=all_gather_symmetric_with_copy_impl,
        fake_impl=all_gather_symmetric_with_copy_fake,
    )


class PyNcclCommunicator:
    def __init__(
        self,
        group: ProcessGroup | StatelessProcessGroup,
        device: int | str | torch.device,
        library_path: str | None = None,
    ):
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the PyNcclCommunicator to. If None,
                it will be bound to f"cuda:{local_rank}".
            library_path: the path to the NCCL library. If None, it will
                use the default library path.
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device.
        """
        if not isinstance(group, StatelessProcessGroup):
            assert dist.is_initialized()
            assert dist.get_backend(group) != dist.Backend.NCCL, (
                "PyNcclCommunicator should be attached to a non-NCCL group."
            )
            # note: this rank is the rank in the group
            self.rank = dist.get_rank(group)
            self.world_size = dist.get_world_size(group)
        else:
            self.rank = group.rank
            self.world_size = group.world_size

        self.group = group

        # Persistent symmetric-memory scratch buffers for reduce_scatter /
        # all_gather. Allocated + NCCL-window-registered once (during eager
        # warmup) and reused as slices, so the capturable hot path never
        # allocates or registers (which would invalidate CUDA graph capture).
        self._symm_scratch_bufs: dict[tuple[str, torch.dtype], torch.Tensor] = {}

        # if world_size == 1, no need to create communicator
        if self.world_size == 1 or envs.VLLM_DISABLE_PYNCCL:
            self.available = False
            self.disabled = True
            return
        try:
            self.nccl = NCCLLibrary(library_path)
        except Exception:
            # disable because of missing NCCL library
            # e.g. in a non-GPU environment
            self.available = False
            self.disabled = True
            return

        self.available = True
        self.disabled = False

        self.nccl_version = self.nccl.ncclGetRawVersion()
        if self.rank == 0:
            # get the unique id from NCCL
            self.unique_id = self.nccl.ncclGetUniqueId()
            logger.info_once("vLLM is using nccl==%s", self.nccl.ncclGetVersion())
        else:
            # construct an empty unique id
            self.unique_id = ncclUniqueId()

        if not isinstance(group, StatelessProcessGroup):
            tensor = torch.ByteTensor(list(self.unique_id.internal))
            ranks = dist.get_process_group_ranks(group)
            # arg `src` in `broadcast` is the global rank
            dist.broadcast(tensor, src=ranks[0], group=group)
            byte_list = tensor.tolist()
            for i, byte in enumerate(byte_list):
                self.unique_id.internal[i] = byte
        else:
            self.unique_id = group.broadcast_obj(self.unique_id, src=0)
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device
        # nccl communicator and stream will use this device
        with torch.accelerator.device_index(device.index):
            self.comm: ncclComm_t = self.nccl.ncclCommInitRank(
                self.world_size, self.unique_id, self.rank
            )

            stream = current_stream()
            # A small all_reduce for warmup.
            data = torch.zeros(1, device=device)
            self.all_reduce(data)
            stream.synchronize()
            del data

    def destroy(self):
        if self.available and not self.disabled:
            # ncclCommAbort can block until all CUDA graphs that
            # captured NCCL ops on this comm are destroyed — and
            # those graphs are released later in this same main-
            # thread teardown, so a direct call here self-deadlocks.
            # Run it in a daemon thread with a timeout: the main
            # thread proceeds, the graphs drop, and the abort returns.
            def _abort():
                with torch.accelerator.device_index(self.device.index):
                    self.nccl.ncclCommAbort(self.comm)

            abort_thread = threading.Thread(target=_abort, daemon=True)
            abort_thread.start()
            abort_thread.join(timeout=5.0)
            self.available = False
            self.disabled = True

    def get_symm_scratch(
        self, name: str, numel: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """A persistent, NCCL-window-registered scratch buffer of >= numel
        elements. (Re)allocated lazily and only when not capturing, so the
        captured hot path reuses an already-registered window."""
        from vllm.distributed.device_communicators.pynccl_allocator import (
            nccl_symm_mem_context,
        )

        key = (name, dtype)
        buf = self._symm_scratch_bufs.get(key)
        if buf is None or buf.numel() < numel:
            assert not torch.cuda.is_current_stream_capturing(), (
                "symmetric-memory scratch must be sized during warmup, "
                "before CUDA graph capture"
            )
            with nccl_symm_mem_context(self):
                buf = torch.empty(numel, dtype=dtype, device=device)
            self._symm_scratch_bufs[key] = buf
        return buf[:numel]

    def all_reduce(
        self,
        in_tensor: torch.Tensor,
        out_tensor: torch.Tensor = None,
        op: ReduceOp = ReduceOp.SUM,
        stream=None,
    ) -> torch.Tensor:
        if self.disabled:
            return None
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert in_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {in_tensor.device}"
        )

        if out_tensor is None:
            out_tensor = torch.empty_like(in_tensor)

        if stream is None:
            stream = current_stream()
        self.nccl.ncclAllReduce(
            buffer_type(in_tensor.data_ptr()),
            buffer_type(out_tensor.data_ptr()),
            in_tensor.numel(),
            ncclDataTypeEnum.from_torch(in_tensor.dtype),
            ncclRedOpTypeEnum.from_torch(op),
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )
        return out_tensor

    def all_gather(
        self, output_tensor: torch.Tensor, input_tensor: torch.Tensor, stream=None
    ):
        if self.disabled:
            return
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )
        if stream is None:
            stream = current_stream()
        self.nccl.ncclAllGather(
            buffer_type(input_tensor.data_ptr()),
            buffer_type(output_tensor.data_ptr()),
            input_tensor.numel(),
            ncclDataTypeEnum.from_torch(input_tensor.dtype),
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def all_gatherv(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        sizes: list[int],
        stream=None,
    ):
        if self.disabled:
            return
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )
        if stream is None:
            stream = current_stream()
        assert output_tensor.shape[0] == sum(sizes)
        split_offset = 0
        self.nccl.ncclGroupStart()
        for root, split_size in enumerate(sizes):
            dst_slice = output_tensor[split_offset : split_offset + split_size]
            self.nccl.ncclBroadcast(
                buffer_type(input_tensor.data_ptr()),
                buffer_type(dst_slice.data_ptr()),
                dst_slice.numel(),
                ncclDataTypeEnum.from_torch(input_tensor.dtype),
                root,
                self.comm,
                cudaStream_t(stream.cuda_stream),
            )
            split_offset += split_size
        self.nccl.ncclGroupEnd()

    def reduce_scatter(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
        stream=None,
    ):
        if self.disabled:
            return
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )
        if stream is None:
            stream = current_stream()
        self.nccl.ncclReduceScatter(
            buffer_type(input_tensor.data_ptr()),
            buffer_type(output_tensor.data_ptr()),
            output_tensor.numel(),
            ncclDataTypeEnum.from_torch(input_tensor.dtype),
            ncclRedOpTypeEnum.from_torch(op),
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def reduce_scatterv(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        sizes: list[int],
        op: ReduceOp = ReduceOp.SUM,
        stream=None,
    ):
        if self.disabled:
            return
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )
        if stream is None:
            stream = current_stream()

        split_offset = 0
        self.nccl.ncclGroupStart()
        for root, split_size in enumerate(sizes):
            chunk = input_tensor[split_offset : split_offset + split_size, ...]
            self.nccl.ncclReduce(
                buffer_type(chunk.data_ptr()),
                buffer_type(output_tensor.data_ptr()),
                chunk.numel(),
                ncclDataTypeEnum.from_torch(input_tensor.dtype),
                ncclRedOpTypeEnum.from_torch(op),
                root,
                self.comm,
                cudaStream_t(stream.cuda_stream),
            )
            split_offset += split_size
        self.nccl.ncclGroupEnd()

    def send(self, tensor: torch.Tensor, dst: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = current_stream()
        if tensor.dtype in [
            torch.float8_e5m2,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2fnuz,
        ]:
            nccl_dtype = ncclDataTypeEnum.from_torch(torch.uint8)
        else:
            nccl_dtype = ncclDataTypeEnum.from_torch(tensor.dtype)
        self.nccl.ncclSend(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            nccl_dtype,
            dst,
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def recv(self, tensor: torch.Tensor, src: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = current_stream()
        if tensor.dtype in [
            torch.float8_e5m2,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2fnuz,
        ]:
            nccl_dtype = ncclDataTypeEnum.from_torch(torch.uint8)
        else:
            nccl_dtype = ncclDataTypeEnum.from_torch(tensor.dtype)
        self.nccl.ncclRecv(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            nccl_dtype,
            src,
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def broadcast(self, tensor: torch.Tensor, src: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = current_stream()
        if src == self.rank:
            sendbuff = buffer_type(tensor.data_ptr())
            # NCCL requires the sender also to have a receive buffer
            recvbuff = buffer_type(tensor.data_ptr())
        else:
            sendbuff = buffer_type()
            recvbuff = buffer_type(tensor.data_ptr())
        self.nccl.ncclBroadcast(
            sendbuff,
            recvbuff,
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            src,
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def group_start(self):
        self.nccl.ncclGroupStart()

    def group_end(self):
        self.nccl.ncclGroupEnd()

    def register_comm_window(self, tensor: torch.Tensor):
        return self.nccl.ncclCommWindowRegister(
            self.comm,
            buffer_type(tensor.data_ptr()),
            tensor.numel() * tensor.element_size(),
            1,
        )

    def register_comm_window_raw(self, ptr: int, size: int):
        return self.nccl.ncclCommWindowRegister(self.comm, buffer_type(ptr), size, 1)

    def deregister_comm_window(self, window):
        return self.nccl.ncclCommWindowDeregister(self.comm, window)

    def batch_isend_irecv(self, p2p_ops: list, stream=None):
        if self.disabled:
            return
        if stream is None:
            stream = current_stream()
        self.group_start()
        for op in p2p_ops:
            if op.op is torch.distributed.isend:
                self.send(op.tensor, op.group_peer, stream)
            elif op.op is torch.distributed.irecv:
                self.recv(op.tensor, op.group_peer, stream)

        self.group_end()
