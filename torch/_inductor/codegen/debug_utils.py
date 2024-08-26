# mypy: allow-untyped-defs
from __future__ import annotations

import functools
import os
from enum import Enum
from typing import List, Optional

from .. import config
from ..virtualized import V
from .common import TensorArg


# AOTI debug printing related configs
class IntermediateValueDebuggingLevel(Enum):
    # OFF: No intermediate tensor value debug info will be printed or saved.
    OFF = 0
    # LEVEL 1: Save all intermediate tensor values to individual `.pt` files. No debug printing will be displayed.
    SAVE = 1
    # LEVEL 2: Print all intermediate tensor values by default to the console.
    DEFAULT_PRINT = 2
    # LEVEL 3: Print selected intermediate tensor values to the console. (specified by the `filtered_kernel_names` env var)
    FILTERED_PRINT = 3


def aot_inductor_debug_intermediate_tensor_value_level() -> Enum:
    if os.environ.get("AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER", "0") == "1":
        return IntermediateValueDebuggingLevel.SAVE
    if os.environ.get("AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER", "0") == "2":
        return IntermediateValueDebuggingLevel.DEFAULT_PRINT
    if os.environ.get("AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER", "0") == "3":
        return IntermediateValueDebuggingLevel.FILTERED_PRINT
    return IntermediateValueDebuggingLevel.OFF


class DebugPrinterManager:
    def __init__(
        self,
        enable_debug_printer: Enum = IntermediateValueDebuggingLevel.OFF,
        args_to_print_or_save: Optional[List[str]] = None,
        kernel_name: str = "",
        kernel=None,
        arg_signatures: Optional[List[type]] = None,
    ):
        self.enable_debug_printer = aot_inductor_debug_intermediate_tensor_value_level()
        if args_to_print_or_save is None:
            args_to_print_or_save = []
        self.args_to_print_or_save = args_to_print_or_save
        self.kernel_name = kernel_name
        self.arg_signatures: Optional[List[type]] = None
        self.kernel = kernel
        self.filtered_kernel_names_to_print = self.get_debug_filtered_kernel_names()

    def __enter__(self):
        if self.enable_debug_printer != IntermediateValueDebuggingLevel.OFF:
            V.graph.all_codegen_kernel_names.add(self.kernel_name)
            # by default save all the tensor value before launch
            self.codegen_intermediate_tensor_value_save(
                self.args_to_print_or_save,
                self.kernel_name,
                before_launch=True,
                arg_signatures=self.arg_signatures,
            )
            if self.enable_debug_printer != IntermediateValueDebuggingLevel.SAVE:
                # not the default save only level, so we want to print the tensor value
                self.codegen_intermediate_tensor_value_printer(
                    self.args_to_print_or_save,
                    self.kernel_name,
                    before_launch=True,
                    arg_signatures=self.arg_signatures,
                )

    def __exit__(self, args_to_print, kernel_name, arg_signatures):
        if self.enable_debug_printer != IntermediateValueDebuggingLevel.OFF:
            # by default save all the tensor value after launch
            self.codegen_intermediate_tensor_value_save(
                self.args_to_print_or_save,
                self.kernel_name,
                before_launch=False,
                arg_signatures=self.arg_signatures,
            )
            if self.enable_debug_printer != IntermediateValueDebuggingLevel.SAVE:
                self.codegen_intermediate_tensor_value_printer(
                    self.args_to_print_or_save,
                    self.kernel_name,
                    before_launch=False,
                    arg_signatures=self.arg_signatures,
                )

    def set_printer_args(
        self,
        args_to_print: List[str],
        kernel_name: str,
        arg_signatures: Optional[List[type]],
        kernel,
    ):
        self.args_to_print_or_save = args_to_print
        self.kernel_name = kernel_name
        self.arg_signatures = arg_signatures
        self.kernel = kernel

    @functools.lru_cache  # noqa: B019
    def get_debug_filtered_kernel_names(self) -> List[str]:
        if config.aot_inductor.filtered_kernel_names is None:
            return []
        return [
            x.strip()
            for x in config.aot_inductor.filtered_kernel_names.lower().split(",")
        ]

    def codegen_intermediate_tensor_value_save(
        self,
        args_to_save,
        kernel_name,
        before_launch=True,
        arg_signatures: Optional[List[type]] = None,
    ) -> None:
        for i, arg in enumerate(args_to_save):
            if arg_signatures is not None and not isinstance(
                arg_signatures[i], TensorArg
            ):
                continue
            launch_prefix = "before_launch" if before_launch else "after_launch"
            if V.graph.cpp_wrapper:
                if config.abi_compatible:
                    V.graph.wrapper_code.writeline(
                        f'aoti_torch_save_tensor_handle({arg}, "{arg}", "{launch_prefix}", "{kernel_name}");'
                    )
                else:
                    # TODO: add non-abi compatible mode debug printing info
                    pass
            else:
                # currently, not cpp wrapper codegen mode not supported.
                pass

    def codegen_intermediate_tensor_value_printer(
        self,
        args_to_print,
        kernel_name,
        before_launch=True,
        arg_signatures: Optional[List[type]] = None,
    ) -> None:
        for i, arg in enumerate(args_to_print):
            if arg_signatures is not None and not isinstance(
                arg_signatures[i], TensorArg
            ):
                continue
            # check the `enable_debug_printer` value. if level 3, check if is the current kernel name in the filtered kernel list
            if (
                self.enable_debug_printer
                == IntermediateValueDebuggingLevel.FILTERED_PRINT
            ):
                assert len(self.filtered_kernel_names_to_print) > 0
                if kernel_name not in self.filtered_kernel_names_to_print:
                    continue

            launch_prefix = "before_launch" if before_launch else "after_launch"
            if V.graph.cpp_wrapper:
                if config.abi_compatible:
                    V.graph.wrapper_code.writeline(
                        f'aoti_torch_print_tensor_handle({arg}, "{launch_prefix} - {kernel_name} - {arg}");'
                    )
                else:
                    # TODO: add non-abi compatible mode debug printing info
                    pass
            else:
                line = f"print('{launch_prefix} - {kernel_name} - {arg}', {arg})"
                V.graph.wrapper_code.writeline(line)
