import os
import argparse
import re
import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

"""
MMShell C++ 源码替换程序
"""

CLANG_DIR = "/home/shenda/dev/clang/clang+llvm-17.0.6-x86_64-linux-gnu-ubuntu-22.04"
CLANG_VERSION = "17"  # clang main version
GCC_VERSION = "11"  # gcc main version
C_INCLUDE_PATH = f"{CLANG_DIR}/lib/clang/{CLANG_VERSION}/include:/usr/include/c++/{GCC_VERSION}:/usr/include/x86_64-linux-gnu/c++/{GCC_VERSION}:/usr/include/c++/{GCC_VERSION}/backward:/usr/local/include:/usr/include/x86_64-linux-gnu:/usr/include"
CPLUS_INCLUDE_PATH = f"{CLANG_DIR}/lib/clang/{CLANG_VERSION}/include:/usr/include/c++/{GCC_VERSION}:/usr/include/x86_64-linux-gnu/c++/{GCC_VERSION}:/usr/include/c++/{GCC_VERSION}/backward:/usr/local/include:/usr/include/x86_64-linux-gnu:/usr/include"
ANALYZER_PATH = "./build/analyzer"

UPDATE_DICT = {
    "AMap": "TJMap",
    "amap": "tjmap",
    "Amap": "TJmap",
    "AMAP": "TJMAP",
    "aMap": "TJMap",
}  # 替换字符串和新字符串的映射关系
NS_STRING_DICT = {**UPDATE_DICT}  # namespace替换字符串和新字符串的映射关系
NS_SKIP_STRING_LIST = []  # namespace忽略替换字符串列表
CLS_STRING_DICT = {**UPDATE_DICT}  # 类型替换字符串和新字符串的映射关系
CLS_SKIP_STRING_LIST = []  # 类型忽略替换字符串列表
MC_STRING_DICT = {
    **NS_STRING_DICT,
    **CLS_STRING_DICT,
}  # 宏替换字符串和新字符串的映射关系
MC_SKIP_STRING_LIST = []  # 宏忽略替换字符串列表
CM_STRING_DICT = {
    **NS_STRING_DICT,
    **CLS_STRING_DICT,
}  # 注释替换字符串和新字符串的映射关系
CM_SKIP_STRING_LIST = []  # 注释忽略替换字符串列表
EXTRA_COMPILE_FLAGS = ""  # 额外的编译选项
SKIP_UPDATE_PATHS = [
    "/build",
    "/ThirdPartyLib",
    "/binaries",
    "/usr",
    "/Qt",
    "/open",
    "__autogen",
]  # C++源文件忽略路径
RM_FILES = ["AMapLicenseListener.h"]  # C++源文件需要删除的文件列表

logging_lock = threading.Lock()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="update_code.log",
    filemode="w",
)


def LOG(level, message):
    with logging_lock:
        if level == "info":
            logging.info(message)
        elif level == "warning":
            logging.warning(message)
        elif level == "error":
            logging.error(message)


def skip_replace_file(file_path):
    for skip_string in SKIP_UPDATE_PATHS:
        if file_path.find(skip_string) != -1:
            # LOG('info', f'跳过文件: {file_path}')
            return True
    return False


class StringUtils:
    """
    StringUtils 类用于处理字符串的跳过和标记操作。

    方法:
      get_skip_list(list, prefix='(', suffix=')'):
        根据传入的字符串列表，为每个字符串添加指定的前缀和后缀，生成一个跳过字符串正则表达式匹配列表。

        参数:
          list (List[str]): 待处理的字符串列表。
          prefix (str): 用于在字符串前添加的前缀，默认为 '('。
          suffix (str): 用于在字符串后添加的后缀，默认为 ')'。

        返回:
          List[str]: 每个字符串添加前后缀后的正则表达式字符串列表。


      mark_string_skip_pos(stmt, skip_list, mark, pos_start=-1, pos_end=-1):
        在给定的字符串 stmt 中使用正则表达式查找跳过列表中的字符串位置，并使用特定的标记替换这些位置。

        参数:
          stmt (str): 需要处理的字符串。
          skip_list (List[str]): 待跳过的字符串列表。
          mark (str): 用于生成替换标记的前缀字符串。
          pos_start (int): 起始搜索位置，默认为 -1 表示从字符串开始处搜索。
          pos_end (int): 结束搜索位置，默认为 -1 表示搜索至字符串末尾。

        返回:
          str: 替换跳过字符串后的新字符串。


      mark_string_pos(stmt, match_string_dict, skip_list, mark, prefix='(', suffix=')', pos_start=-1, pos_end=-1):
        首先对字符串进行跳过字符串的标记，然后在剩余部分中搜索 match_string_dict 所定义的第一个匹配项进行标记替换。

        参数:
          stmt (str): 需要处理的字符串。
          match_string_dict (Dict[str, str]): 包含匹配模式（键）及对应原始字符串（值）的字典。
          skip_list (List[str]): 用于跳过搜索的字符串列表。
          mark (str): 用于生成匹配标记的前缀字符串。
          prefix (str): 定义匹配模式的前缀，默认为 '('。
          suffix (str): 定义匹配模式的后缀，默认为 ')'。
          pos_start (int): 起始搜索位置，默认为 -1 表示从字符串开始处搜索。
          pos_end (int): 结束搜索位置，默认为 -1 表示搜索至字符串末尾。

        返回:
          Tuple[bool, str]:
            bool: 指示是否找到了匹配项。
            str: 替换匹配项标记后的新字符串。


      mark_all_string_pos(stmt, match_string_dict, skip_list, mark, prefix='(', suffix=')', pos_start=-1, pos_end=-1):
        类似于 mark_string_pos 方法，但该方法会对所有匹配项进行标记替换，而非仅第一个匹配项。

        参数:
          stmt (str): 需要处理的字符串。
          match_string_dict (Dict[str, str]): 包含多个匹配模式及相应原始字符串的字典。
          skip_list (List[str]): 用于跳过部分搜索的字符串列表。
          mark (str): 用于生成匹配标记的前缀字符串。
          prefix (str): 定义匹配模式的前缀，默认为 '('。
          suffix (str): 定义匹配模式的后缀，默认为 ')'。
          pos_start (int): 起始搜索位置，默认为 -1 表示从字符串开始处搜索。
          pos_end (int): 结束搜索位置，默认为 -1 表示搜索至字符串末尾。

        返回:
          Tuple[bool, str]:
            bool: 指示是否至少有一个匹配项被标记替换。
            str: 替换所有匹配项标记后的新字符串。


      updat_string(stmt, match_string_dict, skip_list, mark):
        将已被标记替换的占位符恢复为原始的匹配字符串和跳过字符串。

        参数:
          stmt (str): 包含标记占位符的字符串。
          match_string_dict (Dict[str, str]): 包含原始匹配字符串的字典。
          skip_list (List[str]): 包含原始跳过字符串的列表。
          mark (str): 用于生成占位标记的前缀字符串，与之前产生的标记一致。

        返回:
          str: 用原始字符串替换标记占位符后的新字符串。
    """

    def get_skip_list(self, list, prefix="(", suffix=")"):
        skip_list = [prefix + item + suffix for item in list]
        return skip_list

    def mark_string_skip_pos(self, stmt, skip_list, mark, pos_start=-1, pos_end=-1):
        new_stmt = stmt
        skip_idx = 0
        for skip_string in skip_list:
            pattern = re.compile(f"{skip_string}")
            pos = 0
            skip_mark = f"!{mark}{skip_idx}"
            skip_mark_length = len(skip_mark)
            while pos != -1:
                res = pattern.search(new_stmt, pos)
                if res:
                    if pos_start >= 0 and res.start(1) < pos_start:
                        pos = res.end() + 1
                        continue
                    if pos_end >= 0 and pos + res.end(1) - res.start(1) > pos_end:
                        break
                    new_stmt = (
                        new_stmt[: res.start(1)] + skip_mark + new_stmt[res.end(1) :]
                    )
                    pos = res.start(1) + skip_mark_length
                else:
                    pos = -1
            skip_idx += 1
        return new_stmt

    def mark_string_pos(
        self,
        stmt,
        match_string_dict,
        skip_list,
        mark,
        prefix="(",
        suffix=")",
        pos_start=-1,
        pos_end=-1,
    ):
        new_stmt = self.mark_string_skip_pos(stmt, skip_list, mark, pos_start, pos_end)
        found = False
        match_idx = 0
        for match_string in match_string_dict.keys():
            if found:
                break
            pattern = re.compile(f"{prefix}{match_string}{suffix}")
            pos = 0
            match_mark = f"{mark}{match_idx}"
            match_mark_length = len(match_mark)
            while pos != -1:
                if found:
                    break
                res = pattern.search(new_stmt, pos)
                if res:
                    if pos_start >= 0 and res.start(1) < pos_start:
                        pos = res.end() + 1
                        continue
                    if pos_end >= 0 and pos + res.end(1) - res.start(1) > pos_end:
                        break
                    found = True
                    new_stmt = (
                        new_stmt[: res.start(1)] + match_mark + new_stmt[res.end(1) :]
                    )
                    pos = res.start(1) + match_mark_length
                else:
                    pos = -1
            match_idx += 1
        return found, new_stmt

    def mark_all_string_pos(
        self,
        stmt,
        match_string_dict,
        skip_list,
        mark,
        prefix="(",
        suffix=")",
        pos_start=-1,
        pos_end=-1,
    ):
        new_stmt = self.mark_string_skip_pos(stmt, skip_list, mark, pos_start, pos_end)
        found = False
        match_idx = 0
        for match_string in match_string_dict.keys():
            pattern = re.compile(f"{prefix}{match_string}{suffix}")
            pos = 0
            match_mark = f"{mark}{match_idx}"
            match_mark_length = len(match_mark)
            while pos != -1:
                res = pattern.search(new_stmt, pos)
                if res:
                    if pos_start >= 0 and res.start(1) < pos_start:
                        pos = res.end() + 1
                        continue
                    if pos_end >= 0 and pos + res.end(1) - res.start(1) > pos_end:
                        break
                    found = True
                    new_stmt = (
                        new_stmt[: res.start(1)] + match_mark + new_stmt[res.end(1) :]
                    )
                    pos = res.start(1) + match_mark_length
                else:
                    pos = -1
            match_idx += 1
        return found, new_stmt

    def updat_string(self, stmt, match_string_dict, skip_list, mark):
        new_stmt = stmt
        skip_idx = 0
        for value in skip_list:
            new_stmt = new_stmt.replace(f"!{mark}{skip_idx}", value)
            skip_idx += 1
        match_idx = 0
        for value in match_string_dict.values():
            new_stmt = new_stmt.replace(f"{mark}{match_idx}", value)
            match_idx += 1
        return new_stmt


class SourceCodeMatcher:
    """
    类 SourceCodeMatcher 用于对源码中的特定表达式进行匹配、标记和更新操作，
    主要针对 C/C++ 代码中的命名空间、类、宏、注释以及其他表达式进行处理。

    构造函数:
      __init__(parent)
        参数:
          parent (object): 父对象，需包含以下属性，用于指导标记和替换操作：
            - ns_inner: 命名空间的更新映射字典
            - cls_inner: 类名称的更新映射字典
            - ns_skip: 命名空间跳过字典
            - mc_exp_skip: 宏展开时需要跳过的名称字典
            - mc_def_type_update: 用于宏定义更新的类型映射字典
            - mc_def_type_skip: 宏定义跳过类型的字典
            - output_directory: 输出目录的路径
            - input_directory: 输入目录的路径
            - cantidate_rename_files: 用于记录候选重命名文件的字典

    方法:
      match_expression(file_path, file_line, file_line_column, expression_type, stmt, **extra_info)
        功能:
          根据传入的表达式类型 expression_type 和代码语句 stmt，对源代码进行匹配和更新。
          方法会根据表达式的类型（如 Ifndef、MacroExpands、
          MacroDefined、InclusionDirective、Comment、NamespaceDecl、NamedDecl、MemberExpr、CallExpr、
          FunctionDecl、FunctionImpl、UsingDirectiveDecl）采用不同的标记和替换规则。

        参数:
          file_path (str): 当前处理文件的路径。
          file_line (int): 当前代码所在的行数。
          file_line_column (int): 当前代码所在的列数。
          expression_type (str): 表达式类型。
          stmt (str): 需要处理或匹配的源代码语句。
          **extra_info: 附加信息的键值对，例如：
            - macroname: 用于宏操作时指定的宏名称。
            - include_file: 用于#include的文件路径。

        返回:
          tuple: 返回一个二元组 (new_stmt, stmt)，其中：
            - new_stmt (str): 经 StringUtils 标记和更新后得到的新语句；
            - stmt (str): 原始的语句。
          若没有匹配或更新操作，则返回的可能为两个空字符串。

        说明:
          - 本方法内部使用了 StringUtils 类中的方法（如 mark_all_string_pos、mark_string_pos、updat_string）
            对代码中的特定模式进行标记和替换。
          - 根据不同的表达式类型，方法内部逻辑也会调用不同的标记规则和跳过列表，
            以确保只对符合条件的代码段进行更新。
          - 当遇到未知的宏类型或表达式类型时，会通过日志记录错误信息，并返回空字符串。
    """

    def __init__(self, parent):
        self.parent = parent

    def match_expression(
        self,
        file_path,
        file_line,
        file_line_column,
        expression_type,
        stmt,
        **extra_info,
    ):
        """
        函数说明：
          根据传入的表达式类型和代码语句，对代码中的命名空间、类、宏及其它相关部分进行标记和更新；
          主要用于处理 C/C++ 源码中涉及宏、命名空间、类、注释等的逐步替换和调整。

        参数：
          file_path (str): 当前处理的文件路径（未直接在函数中使用，但可能影响其他上下文处理）。
          file_line (int): 当前处理代码所在的行号。
          file_line_column (int): 当前处理代码所在行中的列号。
          expression_type (str): 表达式或宏的类型标识，用于决定后续的处理逻辑；可能包含特殊后缀如 "_M" 或 "_E"。
          stmt (str): 待处理的代码语句字符串。
          **extra_info: 可选的附加参数字典，根据不同操作需要提供额外信息（例如宏名称等）。

        返回值：
          tuple (new_stmt, stmt):
            new_stmt (str): 经过标记和替换后得到的新的语句字符串。
            stmt (str): 原始输入的语句字符串。
          如果处理过程中未能匹配任何规则，或者出现异常，则返回 ("", "")。

        详细说明：
          - 当 expression_type 后缀为 "_M"（宏相关处理）时：
            • 对于 "Ifndef"：通过调用字符串工具类，对语句中指定的宏字符串位置进行标记，进而更新替换。
            • 对于 "MacroExpands"：根据 extra_info 中的 macroname，对宏展开表达式进行匹配查找和替换，
              同时考虑跳过列表中的内容。
            • 对于 "MacroDefined"：判断宏名称是否在预定义更新字典中，并结合命名空间及类的标记规则，
              实现宏定义相关字符串的更新；还会考虑宏跳过和更新的不同规则。
            • 对于 "InclusionDirective"：处理包含指令中的文件路径更新，并在符合条件时记录重命名候选文件。
            • 对于 "Comment"：使用特定规则标记和更新注释字符串中的部分。
            • 对于 "SourceRangeSkipped"：对源代码范围内的命名空间和类进行标记更新。

          - 当 expression_type 后缀为 "_E"（表达式相关处理）时：
            • 如果表达式以 "+" 结尾(NamedDeclMemberExpr)：标记更新命名空间名及类名部分(对<处理以适配模板)。
            • 如果表达式以 "*" 结尾(DeclRefExprTypeLoc)：标记更新类名部分。
            • 对于 "NamespaceDecl"：匹配命名空间声明，进行标记并更新，同时保存更新信息到内部字典。
            • 对于 "NamedDecl" 与 "MemberExpr"：处理类的声明和成员表达式的标记与替换。
            • 对于 "CallExpr"：针对函数调用表达式进行拆分定位，并对定位到的部分进行标记更新。
            • 对于 "FunctionDecl" 或 "FunctionImpl"：从函数声明或实现中提取函数标识部分，
              然后对命名空间和类部分进行标记与替换。
            • 对于 "UsingDirectiveDecl"：处理使用 using namespace 指令，分别匹配不同的标记规则进行更新。

          - 如果 expression_type 不符合预期的格式或未匹配任何处理逻辑，则会记录错误日志，
            并返回空字符串对 ("", "")。
        """
        if len(stmt) == 0:
            return "", ""
        str_utils = StringUtils()
        NS_UPDATE = self.parent.ns_inner
        NS_SKIP = str_utils.get_skip_list(
            NS_SKIP_STRING_LIST
        ) + str_utils.get_skip_list(list(self.parent.ns_skip.keys()), r"\b(", r")::")
        NS_SKIP_STR = NS_SKIP_STRING_LIST + list(self.parent.ns_skip.keys())
        CLS_UPDATE = self.parent.cls_inner
        CLS_SKIP = str_utils.get_skip_list(CLS_SKIP_STRING_LIST)
        CLS_SKIP_STR = CLS_SKIP_STRING_LIST
        MC_SKIP = str_utils.get_skip_list(MC_SKIP_STRING_LIST)
        MC_SKIP_STR = MC_SKIP_STRING_LIST
        CM_SKIP = str_utils.get_skip_list(CM_SKIP_STRING_LIST)
        CM_SKIP_STR = CM_SKIP_STRING_LIST
        if expression_type[-2:] == "_M":
            expression_type = expression_type[:-2]
            if expression_type == "Ifndef":
                found_mc, new_stmt = str_utils.mark_all_string_pos(
                    stmt, MC_STRING_DICT, [], "_MC_"
                )
                if found_mc:
                    new_stmt = str_utils.updat_string(
                        new_stmt, MC_STRING_DICT, [], "_MC_"
                    )
                    return new_stmt, stmt
            elif expression_type == "MacroExpands":
                macroname = extra_info["macroname"]
                if len(macroname) > 0 and macroname not in self.parent.mc_exp_skip:
                    pos = stmt.find("(")
                    found_mc, new_stmt = str_utils.mark_all_string_pos(
                        stmt, MC_STRING_DICT, MC_SKIP, "_MC_", r"(", r")", -1, pos
                    )
                    if found_mc:
                        new_stmt = str_utils.updat_string(
                            new_stmt, MC_STRING_DICT, MC_SKIP_STR, "_MC_"
                        )
                        return new_stmt, stmt
            elif expression_type == "MacroDefined":
                macroname = extra_info["macroname"]
                nscls_dict = {}
                if len(macroname) > 0 and macroname in self.parent.mc_def_type_update:
                    for item in self.parent.mc_def_type_update[macroname]:
                        nscls_dict[item] = item
                skip_list = MC_SKIP
                skip_list_str = MC_SKIP_STR
                if macroname in self.parent.mc_def_type_skip:
                    skip_list += str_utils.get_skip_list(
                        self.parent.mc_def_type_skip[macroname]
                    )
                    skip_list_str += self.parent.mc_def_type_skip[macroname]
                new_stmt = stmt
                _, new_stmt = str_utils.mark_all_string_pos(
                    new_stmt, nscls_dict, skip_list, "_NSCLS_"
                )
                found_mc, new_stmt = str_utils.mark_all_string_pos(
                    new_stmt, MC_STRING_DICT, [], "_MC_"
                )
                new_stmt = str_utils.updat_string(
                    new_stmt, nscls_dict, skip_list_str, "_NSCLS_"
                )
                found_ns, new_stmt = str_utils.mark_all_string_pos(
                    new_stmt, NS_UPDATE, NS_SKIP, "_NS_", r"(", r")\s*::"
                )
                found_cls, new_stmt = str_utils.mark_all_string_pos(
                    new_stmt, CLS_UPDATE, CLS_SKIP, "_CLS_"
                )
                if found_mc or found_ns or found_cls:
                    new_stmt = str_utils.updat_string(
                        new_stmt, MC_STRING_DICT, [], "_MC_"
                    )
                    new_stmt = str_utils.updat_string(
                        new_stmt, NS_UPDATE, NS_SKIP_STR, "_NS_"
                    )
                    new_stmt = str_utils.updat_string(
                        new_stmt, CLS_UPDATE, CLS_SKIP_STR, "_CLS_"
                    )
                    return new_stmt, stmt
            elif expression_type == "InclusionDirective":
                include_file = extra_info["include_file"]
                if skip_replace_file(include_file):
                    return "", ""
                pos = stmt.rfind("/")
                found_mc, new_stmt = str_utils.mark_all_string_pos(
                    stmt, MC_STRING_DICT, [], "_INC_", r"(", r")", pos
                )
                if found_mc:
                    new_stmt = str_utils.updat_string(
                        new_stmt, MC_STRING_DICT, [], "_INC_"
                    )
                    real_file_path = os.path.join(
                        self.parent.output_directory,
                        os.path.relpath(include_file, self.parent.input_directory),
                    )
                    if real_file_path not in self.parent.cantidate_rename_files:
                        dir_name = os.path.dirname(real_file_path)
                        file_name = os.path.basename(real_file_path)
                        _, file_name = str_utils.mark_all_string_pos(
                            file_name, MC_STRING_DICT, [], "_INC_"
                        )
                        file_name = str_utils.updat_string(
                            file_name, MC_STRING_DICT, [], "_INC_"
                        )
                        self.parent.cantidate_rename_files[real_file_path] = (
                            os.path.join(dir_name, file_name)
                        )
                    return new_stmt, stmt
            elif expression_type == "Comment":
                found_cm, new_stmt = str_utils.mark_all_string_pos(
                    stmt, CM_STRING_DICT, CM_SKIP, "_CM_"
                )
                if found_cm:
                    new_stmt = str_utils.updat_string(
                        new_stmt, CM_STRING_DICT, CM_SKIP_STR, "_CM_"
                    )
                    return new_stmt, stmt
            elif expression_type == "SourceRangeSkipped":
                new_stmt = stmt
                found_ns, new_stmt = str_utils.mark_all_string_pos(
                    new_stmt, NS_UPDATE, NS_SKIP, "_NS_", r"(", r")\s*::"
                )
                found_cls, new_stmt = str_utils.mark_all_string_pos(
                    new_stmt, CLS_UPDATE, CLS_SKIP, "_CLS_"
                )
                if found_ns or found_cls:
                    new_stmt = str_utils.updat_string(
                        new_stmt, NS_UPDATE, NS_SKIP_STR, "_NS_"
                    )
                    new_stmt = str_utils.updat_string(
                        new_stmt, CLS_UPDATE, CLS_SKIP_STR, "_CLS_"
                    )
                    return new_stmt, stmt
            else:
                LOG("error", f"未知宏类型: {expression_type}")
        elif expression_type[-2:] == "_E":
            expression_type = expression_type[:-2]
            if expression_type.endswith("+"):
                pos = stmt.find("<")
                if pos != -1:
                    stmt = stmt[: stmt.find("<")]
                new_stmt = stmt
                found_ns, new_stmt = str_utils.mark_all_string_pos(
                    new_stmt, NS_UPDATE, NS_SKIP, "_NS_", r"(", r")\s*::"
                )
                found_cls, new_stmt = str_utils.mark_all_string_pos(
                    new_stmt, CLS_UPDATE, CLS_SKIP, "_CLS_"
                )
                if found_ns or found_cls:
                    new_stmt = str_utils.updat_string(
                        new_stmt, NS_UPDATE, NS_SKIP_STR, "_NS_"
                    )
                    new_stmt = str_utils.updat_string(
                        new_stmt, CLS_UPDATE, CLS_SKIP_STR, "_CLS_"
                    )
                    return new_stmt, stmt
            elif expression_type.endswith("*"):
                found, new_stmt = str_utils.mark_string_pos(
                    stmt, CLS_UPDATE, CLS_SKIP, "_CLS_"
                )
                if found:
                    new_stmt = str_utils.updat_string(
                        new_stmt, CLS_UPDATE, CLS_SKIP_STR, "_CLS_"
                    )
                    return new_stmt, stmt
            elif expression_type == "NamespaceDecl":
                update_dict = NS_STRING_DICT
                skip_list = str_utils.get_skip_list(
                    NS_SKIP_STRING_LIST
                ) + str_utils.get_skip_list(
                    list(self.parent.ns_skip.keys()), r"\b(", r")\b"
                )
                found, new_stmt = str_utils.mark_string_pos(
                    stmt, update_dict, skip_list, "_NS_"
                )
                if found:
                    new_stmt = str_utils.updat_string(
                        new_stmt, update_dict, skip_list, "_NS_"
                    )
                    self.parent.ns_inner[stmt] = new_stmt
                    return new_stmt, stmt
            elif expression_type == "NamedDecl" or expression_type == "MemberExpr":
                found, new_stmt = str_utils.mark_string_pos(
                    stmt, CLS_UPDATE, CLS_SKIP, "_CLS_"
                )
                if found:
                    new_stmt = str_utils.updat_string(
                        new_stmt, CLS_UPDATE, CLS_SKIP_STR, "_CLS_"
                    )
                    return new_stmt, stmt
            elif expression_type == "CallExpr":
                stmt_list = stmt.split("%%")
                if len(stmt_list) != 2:
                    return "", ""
                if len(stmt_list[0]) == 0:
                    return "", ""
                func_pos_start = stmt_list[0].find(stmt_list[1])
                if func_pos_start == -1:
                    return "", ""
                func_pos_end = func_pos_start + len(stmt_list[1])
                stmt_list[0] = stmt_list[0][func_pos_start:func_pos_end]
                found, new_stmt = str_utils.mark_string_pos(
                    stmt_list[0], CLS_UPDATE, CLS_SKIP, "_CLS_"
                )
                if found:
                    new_stmt = str_utils.updat_string(
                        new_stmt, CLS_UPDATE, CLS_SKIP_STR, "_CLS_"
                    )
                    return new_stmt, stmt_list[0]
            elif expression_type == "FunctionDecl" or expression_type == "FunctionImpl":
                function_exp_list = stmt.split("%%")
                function_prefix = ""
                if len(function_exp_list) == 2:
                    pos_prefix = stmt.find(function_exp_list[1])
                    if pos_prefix != -1:
                        function_prefix = stmt[: pos_prefix + len(function_exp_list[1])]
                        stmt = stmt[len(function_prefix) :]
                pos_call = stmt.find("(")
                if pos_call != -1:
                    stmt = stmt[:pos_call]
                new_stmt = stmt
                found_ns, new_stmt = str_utils.mark_all_string_pos(
                    new_stmt, NS_UPDATE, NS_SKIP, "_NS_", r"(", r")\s*::"
                )
                found_cls, new_stmt = str_utils.mark_all_string_pos(
                    new_stmt, CLS_UPDATE, CLS_SKIP, "_CLS_"
                )
                if found_ns or found_cls:
                    new_stmt = str_utils.updat_string(
                        new_stmt, NS_UPDATE, NS_SKIP_STR, "_NS_"
                    )
                    new_stmt = str_utils.updat_string(
                        new_stmt, CLS_UPDATE, CLS_SKIP_STR, "_CLS_"
                    )
                    return new_stmt, stmt
            elif expression_type == "UsingDirectiveDecl":
                namespace_pattern = r"using\s+namespace\s+"
                namespace_pos = re.search(namespace_pattern, stmt)
                found_ns = False
                found_cls = False
                if namespace_pos:
                    new_stmt = stmt
                    found_ns, new_stmt = str_utils.mark_all_string_pos(
                        new_stmt, NS_UPDATE, NS_SKIP, "_NS_"
                    )
                    if found_ns:
                        new_stmt = str_utils.updat_string(
                            new_stmt, NS_UPDATE, NS_SKIP_STR, "_NS_"
                        )
                        return new_stmt, stmt
                else:
                    found_ns, new_stmt = str_utils.mark_all_string_pos(
                        new_stmt, NS_UPDATE, NS_SKIP, "_NS_", r"(", r")\s*::"
                    )
                    found_cls, new_stmt = str_utils.mark_all_string_pos(
                        new_stmt, CLS_UPDATE, CLS_SKIP, "_CLS_"
                    )
                    if found_ns or found_cls:
                        new_stmt = str_utils.updat_string(
                            new_stmt, NS_UPDATE, NS_SKIP_STR, "_NS_"
                        )
                        new_stmt = str_utils.updat_string(
                            new_stmt, CLS_UPDATE, CLS_SKIP_STR, "_CLS_"
                        )
                        return new_stmt, stmt
            else:
                LOG("error", f"未知表达式类型: {expression_type}")
        else:
            LOG("error", f"错误，类型: {expression_type}")
        return "", ""


class SourceCodeUpdateInfo:
    def __init__(self, file_path):
        self.file_path = file_path
        self.update_position = {}


class SourceCodeUpdatePosition:
    """
    SourceCodeUpdatePosition 类用于记录并匹配代码更新时的文件位置和语句转换信息。

    属性:
      file_path (str): 源文件路径。
      file_line (int): 文件中对应的行号。
      file_line_column (int): 文件中该行的起始列位置（索引）。
      expression_type: 表达式类型标识，用于区分不同的更新场景。
      stmt (str): 原始语句，用于进行位置和内容匹配。
      tostmt (str): 目标语句，即更新后应得到的语句。
      extra_info (dict): 额外信息字典，用于传递其他自定义参数。

    方法:
      __init__(file_path, file_line, file_line_column, expression_type, stmt, tostmt, **extra_info):
        构造函数，用于初始化 SourceCodeUpdatePosition 实例，设置文件位置、表达式类型、原始语句、目标语句以及其他附加信息。

      match(file_path, file_line, file_line_column, expression_type, stmt, tostmt, **extra_info):
        检查给定的新语句和位置信息是否与当前记录匹配。

        匹配逻辑:
          1. 仅在当前对象记录的行号与输入行号相同时开始匹配。
          2. 根据原始和新语句的起始列和长度，计算出各自的结束位置，判断两者的区域是否重叠。
          3. 如果新语句完全处在原始语句内且长度较短,
             - 通过全局匹配器 (g_matcher) 进一步检查局部子语句是否匹配，
             - 若匹配成功且转换后的子语句与目标语句相同，则返回 True。
          4. 如果新语句范围包含原始语句且长度更长,
             - 根据位置提取新语句中的子串，
             - 使用全局匹配器 (g_matcher) 验证该子串在转换后是否与目标语句一致，
             - 如匹配成功，则将原始语句和目标语句置为空。
          5. 对于其他部分重叠情况，直接返回 True。
          6. 若所有条件均不满足，则返回 False。
    """

    def __init__(
        self,
        file_path,
        file_line,
        file_line_column,
        expression_type,
        stmt,
        tostmt,
        **extra_info,
    ):
        self.file_path = file_path
        self.file_line = file_line
        self.file_line_column = file_line_column
        self.expression_type = expression_type
        self.stmt = stmt
        self.tostmt = tostmt
        self.extra_info = extra_info

    def match(
        self,
        file_path,
        file_line,
        file_line_column,
        expression_type,
        stmt,
        tostmt,
        **extra_info,
    ):
        if self.file_line == file_line:
            global g_matcher
            current_start = self.file_line_column
            current_length = len(self.stmt)
            current_end = self.file_line_column + current_length
            new_start = file_line_column
            new_length = len(stmt)
            new_end = file_line_column + new_length
            if new_start < current_end and new_end > current_start:
                if (
                    new_start >= current_start
                    and new_end <= current_end
                    and new_length < current_length
                ):
                    pos_start = new_start - current_start
                    pos_end = pos_start + new_length
                    sub_stmt = self.stmt[pos_start:pos_end]
                    new_sub_stmt, _ = g_matcher.match_expression(
                        self.file_path,
                        self.file_line,
                        pos_start,
                        self.expression_type,
                        sub_stmt,
                        **self.extra_info,
                    )
                    if sub_stmt == stmt:
                        if new_sub_stmt == tostmt:
                            return True
                        return None
                elif (
                    new_start <= current_start
                    and new_end >= current_end
                    and new_length > current_length
                ):
                    pos_start = current_start - new_start
                    pos_end = pos_start + current_length
                    sub_stmt = stmt[pos_start:pos_end]
                    new_sub_stmt, _ = g_matcher.match_expression(
                        file_path,
                        file_line,
                        pos_start,
                        expression_type,
                        sub_stmt,
                        **extra_info,
                    )
                    if sub_stmt == self.stmt:
                        if new_sub_stmt == self.tostmt:
                            self.stmt = self.tostmt = ""
                        else:
                            return None
                else:
                    return True
        return False


class SourceCodeUpdater:
    """
    类 SourceCodeUpdater
    该类用于根据编译配置和静态分析结果自动化更新源代码文件，支持批量复制、分析、替换和重命名操作。
    主要功能:
      1. 解析编译命令配置文件，获取每个源文件对应的编译参数。
      2. 检查输入、输出和日志目录的合法性，进行必要的路径转换。
      3. 调用外部静态分析工具分析源文件，并通过分析日志匹配待替换的代码位置。
      4. 校验匹配位置的代码内容，确保替换操作精确无误。
      5. 对复制到输出目录的源文件按照匹配的信息进行代码内容的替换更新。
      6. 支持多线程并发执行分析操作，提高大规模代码处理的效率。
      7. 最后对更新后的文件进行必要的重命名操作，完成整个源代码更新流程。
    主要方法:
      __init__(self, compile_commands_json, input_directory, output_directory, log_directory):
        初始化编译配置、输入/输出和日志目录，并设置相关数据结构与线程锁。
      parse_compile_commands(self):
        从配置文件中解析编译命令，收集源文件与其编译参数的映射关系，校验文件有效性。
      exec_analyzer(self, cmd, files, skip=True):
        调用外部分析工具对指定源文件进行代码分析，生成日志文件供后续处理，支持多线程执行。
      match_update_position(self, file_path, file_line, file_line_column, expression_type, stmt, tostmt, **extra_info):
        校验并匹配给定位置的代码表达式，判断是否需要替换，记录替换信息并处理冲突情况。
      analyze_source_files_MacroDefExpIfndefInclusionCommentSkip(self, reuse_analyzed_result=False):
        分析宏定义、宏展开、条件包含和注释相关的日志，分别收集更新或跳过更新的信息。
      check_expression_log_entry(self, log_entry, log_entry_type, append_type=''):
        检查单条日志记录，验证并匹配表达式内容，计算替换位置后记录相应的替换指令。
      analyze_source_files_skip(self, analyze_type, reuse_analyzed_result=True):
        分析需要跳过处理的日志信息，如命名空间声明或宏定义，记录无需替换的代码部分。
      analyze_source_files(self, analyze_type, append_type='', reuse_analyzed_result=True):
        根据指定的分析类型处理日志记录，将匹配的表达式转换为待替换操作，并记录更新位置。
      replace_in_source_files(self):
        根据已收集的替换位置信息，对复制后的源代码文件逐行更新，实现精确替换。
      copy_source_files(self):
        将输入目录下的所有源代码文件复制到输出目录，通过路径相对关系保证文件结构一致。
      rename_updated_files(self):
        对更新后的文件执行重命名操作，确保文件名的一致性或符合特定需求。
      process_source_files(self):
        整合复制、日志分析、代码替换与文件重命名等步骤，完成整个源代码更新的自动化流程。
    """

    def __init__(
        self, compile_commands_json, input_directory, output_directory, log_directory
    ):
        self.compile_commands_json = compile_commands_json
        self.source_files = {}
        self.parse_compile_commands()
        self.input_directory = os.path.realpath(input_directory)
        self.output_directory = os.path.realpath(output_directory)
        self.log_directory = os.path.realpath(log_directory)
        if not os.path.exists(self.compile_commands_json):
            LOG("error", f"错误: 编译配置 '{self.compile_commands_json}' 不存在")
            print(
                f"错误: 编译配置 '{self.compile_commands_json}' 不存在", file=sys.stderr
            )
            exit(1)
        if not os.path.exists(self.input_directory):
            LOG("error", f"错误: 输入目录 '{self.input_directory}' 不存在")
            print(f"错误: 输入目录 '{self.input_directory}' 不存在", file=sys.stderr)
            exit(1)
        if os.path.exists(self.output_directory):
            LOG("warning", f"输出目录 '{self.output_directory}' 已存在")
        if os.path.exists(self.log_directory):
            LOG("warning", f"日志目录 '{self.log_directory}' 已存在")
        self.cantidate_replace_files = {}
        self.cantidate_rename_files = {}
        self.lock = threading.Lock()
        self.ns_inner = {}
        self.cls_inner = {**CLS_STRING_DICT}
        self.ns_skip = {}
        self.mc_def_type_update = {}
        self.mc_def_type_skip = {}
        self.mc_exp_skip = {}

    def parse_compile_commands(self):
        with open(self.compile_commands_json, "r") as f:
            compile_commands = json.load(f)
        for entry in compile_commands:
            command = entry["command"]
            compile_commands = command.split(" -o ")[0]
            source_file = os.path.abspath(entry["file"])
            if skip_replace_file(source_file):
                continue
            self.source_files[source_file] = compile_commands
        if len(self.source_files) == 0:
            LOG("error", f"错误: 没有从{self.compile_commands_json}找到任何编译配置")
            print(
                f"错误: 没有从{self.compile_commands_json}找到任何编译配置",
                file=sys.stderr,
            )
            exit(1)

    def exec_analyzer(self, cmd, files, skip=True):
        # 设置分析日志文件的前缀
        analyzer_log_file_prefix = f"{self.log_directory}/analyze_{cmd}_"
        # 如果需要跳过分析，则直接返回
        if skip == True:
            return
        # 删除已有的分析日志文件
        result = os.system(f"rm -fr {analyzer_log_file_prefix}*log")
        if result != 0:
            LOG("error", f"错误: 返回值: {result} 删除{analyzer_log_file_prefix}*log")
            print(
                f"错误: 返回值: {result} 删除{analyzer_log_file_prefix}*log",
                file=sys.stderr,
            )
            exit(1)
        # 设置环境变量：C_INCLUDE_PATH、CPLUS_INCLUDE_PATH 和 ANALYZE_SKIP_PATH
        os.environ["C_INCLUDE_PATH"] = C_INCLUDE_PATH
        os.environ["CPLUS_INCLUDE_PATH"] = CPLUS_INCLUDE_PATH
        os.environ["ANALYZE_SKIP_PATH"] = ":".join(SKIP_UPDATE_PATHS)

        # 内部函数：使用线程池执行分析任务
        def exec_analyzer_with_thread_pool(parent, analyzer, cmd, files, pool_size=1):
            # 内部函数：对单个文件执行分析命令
            def exec_command(
                analyzer, file_path, analyzer_log_file_prefix, cmd, compile_commands
            ):
                LOG("info", f"分析 {cmd}:{file_path}")
                # 为当前文件生成日志文件名
                analyzer_log_file = (
                    f"{analyzer_log_file_prefix}{os.path.basename(file_path)}.log"
                )
                # 将文件标识写入日志文件，并调用分析器对文件进行分析
                result = os.system(
                    f'echo "#\n#\n# {file_path}\n#\n#" >> {analyzer_log_file} && ANALYZE_CMD="{cmd}|" {analyzer} {file_path} -- {compile_commands} -Wno-everything >> {analyzer_log_file}'
                )
                if result != 0:
                    LOG(
                        "error", f"错误: 分析源文件时出错，返回值: {result} {file_path}"
                    )
                    print(
                        f"错误: 分析源文件时出错，返回值: {result} {file_path}",
                        file=sys.stderr,
                    )
                    exit(1)

            # 使用线程池并发执行分析任务
            with ThreadPoolExecutor(max_workers=pool_size) as executor:
                futures = []
                index = 0
                # 遍历每个需要分析的文件
                for file_path in files:
                    futures.append(
                        executor.submit(
                            exec_command,
                            analyzer,
                            file_path,
                            f"{analyzer_log_file_prefix}{index:04d}_",
                            cmd,
                            parent.source_files[file_path] + EXTRA_COMPILE_FLAGS,
                        )
                    )
                    index += 1
                # 等待所有线程任务执行完毕
                for future in futures:
                    future.result()

        # 调用线程池函数，设置最大线程数为10
        exec_analyzer_with_thread_pool(self, ANALYZER_PATH, cmd, files, pool_size=10)

    def match_update_position(
        self,
        file_path,
        file_line,
        file_line_column,
        expression_type,
        stmt,
        tostmt,
        **extra_info,
    ):
        """
        匹配并更新指定文件中源码的位置。

        此方法用于在文件中定位给定的语句(stmt)位置，并验证实际内容是否与预期一致。如果内容匹配，
        则在候选替换记录中查找是否已有相同位置的更新操作。若操作冲突或存在错误，则记录相应日志信息，
        并返回 True 表示存在问题；否则返回 False 表示匹配成功且无冲突。

        参数：
          file_path (str): 文件的路径。
          file_line (int): 源代码所在的行号（基于1的索引）。
          file_line_column (int): 源代码所在行的列号（基于1的索引）。
          expression_type (str): 表达式类型，用于描述操作类型。
          stmt (str): 预期在指定位置出现的原始语句内容。
          tostmt (str): 替换为的新语句内容。
          **extra_info: 可选的额外信息参数，用于支持更多扩展功能。

        返回：
          bool: 如果存在匹配错误、内容不一致或替换操作冲突，则返回 True；否则返回 False。

        描述：
          1. 首先，在线程锁保护下，检查文件是否已存在于候选替换记录中，
             如不存在则创建相应的 SourceCodeUpdateInfo 对象。
          2. 内部定义的辅助函数 match_stmt_at_position 用于读取文件中指定位置的内容，
             将其与预期的语句内容进行逐字符比较，若不匹配则返回错误信息。
          3. 若文件内容不匹配，则记录警告日志，并返回 True。
          4. 若内容匹配，则遍历该文件中所有已存在的更新位置，
             依次调用其 match 方法检测新替换操作和现有操作是否存在冲突。
          5. 若发现冲突或错误替换，则记录错误日志，并返回 True。
          6. 若无任何问题，则返回 False，表示更新位置匹配且可安全执行替换操作。

        """
        with self.lock:
            if file_path not in self.cantidate_replace_files:
                self.cantidate_replace_files[file_path] = SourceCodeUpdateInfo(
                    file_path
                )

        def match_stmt_at_position(file_path, file_line, file_line_column, stmt):
            with open(file_path, "r") as f:
                lines = f.readlines()
            if file_line < 1 or file_line > len(lines):
                return f"bad file line {file_path}:{file_line}"
            line = lines[file_line - 1]
            col_index = file_line_column - 1
            if col_index < 0 or col_index > len(line):
                return (
                    f"bad file line column {file_path}:{file_line}:{file_line_column}"
                )
            file_stmt = ""
            multi_lines = stmt.splitlines(keepends=True)
            if len(multi_lines) > 1:
                file_stmt += line[col_index:]
                index = 1
                for _ in range(1, len(multi_lines) - 1):
                    file_stmt += lines[file_line - 1 + index]
                    index += 1
                file_stmt += lines[file_line - 1 + index][: len(multi_lines[-1])]
            else:
                file_stmt = line[col_index : col_index + len(multi_lines[-1])]
            if file_stmt != stmt:
                return f"content not match {file_path}:{file_line}:{file_line_column} [[{stmt}]] != [[{file_stmt}]]"
            return None

        match_error = match_stmt_at_position(
            file_path, file_line, file_line_column, stmt
        )
        if match_error:
            LOG("warning", f"忽略不匹配的源码内容 {expression_type} {match_error}")
            return True
        with self.lock:
            for pos in self.cantidate_replace_files[file_path].update_position.values():
                result = pos.match(
                    file_path,
                    file_line,
                    file_line_column,
                    expression_type,
                    stmt,
                    tostmt,
                    **extra_info,
                )
                if result is None:
                    LOG("error", f"错误替换 {file_path}")
                    LOG(
                        "error",
                        f"错误替换 {expression_type} {file_line}:{file_line_column} [[{stmt}]] -> [[{tostmt}]]",
                    )
                    LOG(
                        "error",
                        f"已存在替换 {pos.expression_type} {pos.file_line}:{pos.file_line_column} [[{pos.stmt}]] -> [[{pos.tostmt}]]",
                    )
                    return True
                elif result:
                    return True
        return False

    def analyze_source_files_MacroDefExpIfndefInclusionCommentSkip(
        self, reuse_analyzed_result=False
    ):
        global g_matcher
        # 执行外部分析器，处理类型为 "MacroDefExpIfndefInclusionCommentSkip" 的分析任务
        self.exec_analyzer(
            "MacroDefExpIfndefInclusionCommentSkip",
            self.source_files.keys(),
            reuse_analyzed_result,
        )
        # 获取日志目录下所有日志文件的完整路径
        log_files = [
            os.path.join(self.log_directory, f)
            for f in os.listdir(self.log_directory)
            if os.path.isfile(os.path.join(self.log_directory, f))
        ]
        # 第一轮：读取日志文件并处理宏中使用到的C++类型（类、enum等类型）信息
        for analyze_log_file in log_files:
            if not "MacroDefExpIfndefInclusionCommentSkip" in analyze_log_file:
                continue
            with open(analyze_log_file, "r") as analyze_log_file_content:
                line_no = 0
                for line in analyze_log_file_content:
                    line_no += 1
                    if line.startswith("#"):
                        continue
                    try:
                        log_entry = json.loads(line)
                        macrotype = log_entry["type"]
                        # 如果类型为 'DeclRefExprTypeLoc'，则处理宏定义相关内容
                        if macrotype == "DeclRefExprTypeLoc":
                            stmt = log_entry["stmt"]
                            macro_list = stmt.split("%%")
                            if len(macro_list) != 2:
                                continue
                            # 如果宏名称以 '!' 开头，则该宏为跳过更新类型
                            if macro_list[1].startswith("!"):
                                macroname = macro_list[1][1:]
                                if macroname not in self.mc_def_type_skip:
                                    self.mc_def_type_skip[macroname] = []
                                self.mc_def_type_skip[macroname].append(macro_list[0])
                            else:
                                # 否则，记录用于更新的宏定义
                                macroname = macro_list[1]
                                if macroname not in self.mc_def_type_update:
                                    self.mc_def_type_update[macroname] = []
                                self.mc_def_type_update[macroname].append(macro_list[0])
                    except json.JSONDecodeError as e:
                        LOG(
                            "error",
                            f"错误: 解析宏分析日志时出错: {analyze_log_file} {line_no} {e} {line}",
                        )
                        print(
                            f"错误: 解析宏分析日志时出错: {analyze_log_file} {line_no} {e} {line}",
                            file=sys.stderr,
                        )
                        exit(1)
        # 第二轮：读取日志文件并处理各种宏的更新信息
        for analyze_log_file in log_files:
            if not "MacroDefExpIfndefInclusionCommentSkip" in analyze_log_file:
                continue
            with open(analyze_log_file, "r") as analyze_log_file_content:
                line_no = 0
                for line in analyze_log_file_content:
                    line_no += 1
                    if line.startswith("#"):
                        continue
                    try:
                        log_entry = json.loads(line)
                        macrotype = log_entry["type"]
                        # 处理 MacroExpands, MacroDefined 和 Ifndef 类型，它们需要追加 '_M'
                        if (
                            macrotype == "MacroExpands"
                            or macrotype == "MacroDefined"
                            or macrotype == "Ifndef"
                        ):
                            macrotype += "_M"
                            file_path = log_entry["file"]
                            file_line = log_entry["line"]
                            file_line_column = log_entry["column"]
                            macroname = log_entry["macroname"]
                            macrostmt = log_entry["macrostmt"]
                            # 调用匹配器对宏表达式进行匹配处理
                            new_macrostmt, _ = g_matcher.match_expression(
                                file_path,
                                file_line,
                                file_line_column,
                                macrotype,
                                macrostmt,
                                macroname=macroname,
                            )
                            # 如果匹配结果非空，则记录更新位置
                            if len(new_macrostmt) > 0:
                                if not self.match_update_position(
                                    file_path,
                                    file_line,
                                    file_line_column,
                                    macrotype,
                                    macrostmt,
                                    new_macrostmt,
                                    macroname=macroname,
                                ):
                                    with self.lock:
                                        self.cantidate_replace_files[
                                            file_path
                                        ].update_position[
                                            (file_line, file_line_column)
                                        ] = SourceCodeUpdatePosition(
                                            file_path,
                                            file_line,
                                            file_line_column,
                                            macrotype,
                                            macrostmt,
                                            new_macrostmt,
                                            macroname=macroname,
                                        )
                        # 处理包含指令表达式
                        elif macrotype == "InclusionDirective":
                            macrotype += "_M"
                            file_path = log_entry["file"]
                            file_line = log_entry["line"]
                            file_line_column = log_entry["column"]
                            include_file_path = log_entry["includefilepath"]
                            stmt = log_entry["stmt"]
                            # 调用匹配器处理包含指令
                            new_stmt, _ = g_matcher.match_expression(
                                file_path,
                                file_line,
                                file_line_column,
                                macrotype,
                                stmt,
                                include_file=include_file_path,
                            )
                            if len(new_stmt) > 0:
                                if not self.match_update_position(
                                    file_path,
                                    file_line,
                                    file_line_column,
                                    macrotype,
                                    stmt,
                                    new_stmt,
                                    include_file=include_file_path,
                                ):
                                    with self.lock:
                                        self.cantidate_replace_files[
                                            file_path
                                        ].update_position[
                                            (file_line, file_line_column)
                                        ] = SourceCodeUpdatePosition(
                                            file_path,
                                            file_line,
                                            file_line_column,
                                            macrotype,
                                            stmt,
                                            new_stmt,
                                            include_file=include_file_path,
                                        )
                        # 处理注释表达式
                        elif macrotype == "Comment":
                            macrotype += "_M"
                            file_path = log_entry["file"]
                            file_line = log_entry["line"]
                            file_line_column = log_entry["column"]
                            stmt = log_entry["stmt"]
                            # 调用匹配器处理注释表达式
                            new_stmt, _ = g_matcher.match_expression(
                                file_path, file_line, file_line_column, macrotype, stmt
                            )
                            if len(new_stmt) > 0:
                                if not self.match_update_position(
                                    file_path,
                                    file_line,
                                    file_line_column,
                                    macrotype,
                                    stmt,
                                    new_stmt,
                                ):
                                    with self.lock:
                                        self.cantidate_replace_files[
                                            file_path
                                        ].update_position[
                                            (file_line, file_line_column)
                                        ] = SourceCodeUpdatePosition(
                                            file_path,
                                            file_line,
                                            file_line_column,
                                            macrotype,
                                            stmt,
                                            new_stmt,
                                        )
                        # 处理源代码中被跳过的范围表达式
                        elif macrotype == "SourceRangeSkipped":
                            macrotype += "_M"
                            file_path = log_entry["file"]
                            file_line = log_entry["line"]
                            file_line_column = log_entry["column"]
                            stmt = log_entry["stmt"]
                            # 调用匹配器处理被跳过的代码范围
                            new_stmt, _ = g_matcher.match_expression(
                                file_path, file_line, file_line_column, macrotype, stmt
                            )
                            if len(new_stmt) > 0:
                                if not self.match_update_position(
                                    file_path,
                                    file_line,
                                    file_line_column,
                                    macrotype,
                                    stmt,
                                    new_stmt,
                                ):
                                    with self.lock:
                                        self.cantidate_replace_files[
                                            file_path
                                        ].update_position[
                                            (file_line, file_line_column)
                                        ] = SourceCodeUpdatePosition(
                                            file_path,
                                            file_line,
                                            file_line_column,
                                            macrotype,
                                            stmt,
                                            new_stmt,
                                        )
                    except json.JSONDecodeError as e:
                        LOG(
                            "error",
                            f"错误: 解析宏分析日志时出错: {analyze_log_file} {line_no} {e} {line}",
                        )
                        print(
                            f"错误: 解析宏分析日志时出错: {analyze_log_file} {line_no} {e} {line}",
                            file=sys.stderr,
                        )
                        exit(1)

    def check_expression_log_entry(self, log_entry, log_entry_type, append_type=""):
        # 如果 log_entry 中的类型不等于指定类型，直接返回
        if log_entry["type"] != log_entry_type:
            return
        file_path = log_entry["file"]
        # 如果文件路径为空，直接返回
        if len(file_path) == 0:
            return
        # 如果文件路径符合跳过条件，则返回
        if skip_replace_file(file_path):
            return
        file_line = log_entry["line"]
        file_line_column = log_entry["column"]
        stmt = log_entry["stmt"]
        # 如果语句为空，则返回
        if len(stmt) == 0:
            return
        expression_type = log_entry["exptype"]
        # 拼接附加类型标识，并添加 _E 后缀
        expression_type += f"{append_type}_E"
        global g_matcher
        # 调用匹配函数进行表达式匹配
        new_stmt, adj_stmt = g_matcher.match_expression(
            file_path, file_line, file_line_column, expression_type, stmt
        )
        # 如果匹配后得到的新语句不为空，继续处理
        if len(new_stmt) > 0:
            multi_lines = False
            # 如果调整后的语句开头有换行字符，则认为是多行表达式
            while adj_stmt[0] in ["\n", "\r"]:
                multi_lines = True
                # 去掉前导的换行符
                adj_stmt = adj_stmt.lstrip("\r\n")
                new_stmt = new_stmt.lstrip("\r\n")
                # 文件行号增加，列号重置为1
                file_line += 1
                file_line_column = 1
            # 如果不是多行语句，则查找调整后语句的位置，并更新列号
            if not multi_lines:
                pos = stmt.find(adj_stmt)
                if pos == -1:
                    LOG(
                        "error",
                        f'表达式匹配错误 {log_entry["file"]} {log_entry["line"]} {log_entry["column"]} {log_entry["stmt"]} {log_entry["exptype"]}',
                    )
                    return
                file_line_column += pos
            # 使用调整后的语句进行替换操作
            stmt = adj_stmt
            # 如果当前位置没有发生更新，则记录更新位置
            if not self.match_update_position(
                file_path, file_line, file_line_column, expression_type, stmt, new_stmt
            ):
                with self.lock:
                    self.cantidate_replace_files[file_path].update_position[
                        (file_line, file_line_column)
                    ] = SourceCodeUpdatePosition(
                        file_path,
                        file_line,
                        file_line_column,
                        expression_type,
                        stmt,
                        new_stmt,
                    )

    def analyze_source_files_skip(self, analyze_type, reuse_analyzed_result=True):
        # 调用分析器执行指定类型的分析任务
        self.exec_analyzer(
            f"{analyze_type}", self.source_files.keys(), reuse_analyzed_result
        )
        # 获取日志目录下所有日志文件的完整路径
        log_files = [
            os.path.join(self.log_directory, f)
            for f in os.listdir(self.log_directory)
            if os.path.isfile(os.path.join(self.log_directory, f))
        ]
        # 遍历所有日志文件
        for analyze_log_file in log_files:
            # 过滤不包含指定分析类型名称的日志文件
            if f"{analyze_type}" not in analyze_log_file:
                continue
            # 打开当前日志文件读取内容
            with open(analyze_log_file, "r") as analyze_log_file_content:
                line_no = 0
                # 按行读取日志文件
                for line in analyze_log_file_content:
                    line_no += 1
                    # 跳过空行或以 '#' 开头的注释行
                    if len(line) == 0 or line.startswith("#"):
                        continue
                    try:
                        # 将日志行解析为 JSON 对象
                        log_entry = json.loads(line)
                        stmt = log_entry["stmt"]
                        # 如果语句非空，则根据分析类型进行相应的处理
                        if len(stmt) > 0:
                            if analyze_type == "SkipNamespaceDecl":
                                # 对 SkipNamespaceDecl 类型，记录命名空间跳过语句
                                self.ns_skip[stmt] = stmt
                            elif analyze_type == "SkipMacroDef":
                                # 对 SkipMacroDef 类型，记录宏定义跳过语句
                                self.mc_exp_skip[stmt] = stmt
                    except json.JSONDecodeError as e:
                        # 解析 JSON 失败，记录错误日志并输出错误信息
                        LOG(
                            "error",
                            f"错误: 解析 {analyze_log_file} 第 {line_no} 行出错: {e} {line}",
                        )
                        print(
                            f"错误: 解析 {analyze_log_file} 第 {line_no} 行出错: {e} {line}",
                            file=sys.stderr,
                        )
                        exit(1)
        # 根据分析类型输出相应的跳过记录日志
        if analyze_type == "SkipNamespaceDecl":
            LOG("info", f"命名空间跳过列表: {self.ns_skip}")
        elif analyze_type == "SkipMacroDef":
            LOG("info", f"宏跳过列表: {self.mc_exp_skip}")

    def analyze_source_files(
        self, analyze_type, append_type="", reuse_analyzed_result=True
    ):
        """
        分析源代码文件，执行代码分析，并检查生成的日志条目。

        参数：
          analyze_type (str): 指定当前要执行的分析器类型，用于确定执行逻辑及日志文件的筛选。
          append_type (str, 可选): 附加的类型信息，用于在检查日志条目时进行额外处理，默认为空字符串。
          reuse_analyzed_result (bool, 可选): 指定是否重用之前分析的结果，默认为 True。

        流程：
          1. 通过调用 exec_analyzer 方法，根据 analyze_type 参数对所有源文件执行分析。
          2. 遍历日志目录下的所有日志文件，筛选出文件名中包含 analyze_type 的日志文件。
          3. 对筛选出的每个日志文件，逐行读取内容：
             - 忽略空行或以 '#' 开头的注释行。
             - 尝试将每一行解析为 JSON 格式，并调用 check_expression_log_entry 进行日志条目检查。
          4. 如果 JSON 解析过程中出现错误，记录错误信息，打印错误到标准错误输出，并终止程序执行（exit(1)）。

        异常：
          如果在解析日志行时触发 JSONDecodeError，将输出详细的错误信息及相关行号，并退出程序。
        """
        global g_matcher
        self.exec_analyzer(
            f"{analyze_type}", self.source_files.keys(), reuse_analyzed_result
        )
        log_files = [
            os.path.join(self.log_directory, f)
            for f in os.listdir(self.log_directory)
            if os.path.isfile(os.path.join(self.log_directory, f))
        ]
        for analyze_log_file in log_files:
            if not f"{analyze_type}" in analyze_log_file:
                continue
            with open(analyze_log_file, "r") as analyze_log_file_content:
                line_no = 0
                for line in analyze_log_file_content:
                    line_no += 1
                    if len(line) == 0 or line.startswith("#"):
                        continue
                    try:
                        log_entry = json.loads(line)
                        self.check_expression_log_entry(
                            log_entry, analyze_type, append_type
                        )
                    except json.JSONDecodeError as e:
                        LOG(
                            "error",
                            f"错误: 解析{analyze_log_file} {line_no}行出错: {e} {line}",
                        )
                        print(
                            f"错误: 解析{analyze_log_file} {line_no}行出错: {e} {line}",
                            file=sys.stderr,
                        )
                        exit(1)

    def replace_in_source_files(self):
        """
        替换源代码文件中的指定内容。

        该方法的主要工作流程如下：
        1. 遍历存储在 self.cantidate_replace_files 中待处理的源文件及其更新信息；
        2. 对于每个源文件，首先判断是否应当跳过更新（通过调用 skip_replace_file 函数），
          如果是则直接跳过；
        3. 根据输入目录和输出目录的对应关系，生成目标文件的完整路径，并将原始文件复制到目标位置；
        4. 从源文件中读取全部行，捕获读取过程中的异常（如文件读取失败），
          如果发生错误则输出错误信息并跳过当前文件；
        5. 如果在更新信息中存在更新位置（update_info.update_position）：
          - 将这些更新位置按文件行号及列号排序，确保按顺序进行替换操作；
          - 针对每个更新位置，检查原始语句（stmt）与目标语句（tostmt）的差异，
            若两者一致，则不进行替换；
          - 如果目标语句为多行，则额外处理相应的合并逻辑；
          - 根据更新位置的行号及列号，计算实际替换的位置，
            使用新语句替换原有语句，并更新行内容；
          - 同时调整替换偏移量以弥补语句长度的变化；
          - 日志记录每次替换操作，详细描述了表达式类型、位置、原始及更新后的语句内容；
        6. 创建目标文件目录（如不存在），并将更新后的内容写入目标文件。

        返回：
           无返回值，直接将结果写入目标文件。
        """
        for source_file, update_info in self.cantidate_replace_files.items():
            if skip_replace_file(source_file):
                continue
            output_source_file = os.path.join(
                self.output_directory,
                os.path.relpath(source_file, self.input_directory),
            )
            LOG("info", f"from file: {source_file}")
            LOG("info", f"to file: {output_source_file}")
            os.system(f"cp {source_file} {output_source_file}")
            lines = []
            with open(source_file, "r") as input_source:
                try:
                    lines = input_source.readlines()
                except Exception as e:
                    print(f"Error reading file {source_file}: {e}")
                    continue
                if len(update_info.update_position) > 0:
                    sorted_positions = sorted(
                        update_info.update_position.values(),
                        key=lambda pos: (pos.file_line, pos.file_line_column),
                    )
                    replace_line_number = -1
                    replace_diff = 0
                    line_content = ""
                    for update_position in sorted_positions:
                        if replace_line_number != update_position.file_line - 1:
                            replace_line_number = update_position.file_line - 1
                            line_content = lines[replace_line_number]
                            replace_diff = 0
                        stmt = update_position.stmt
                        new_stmt = update_position.tostmt
                        if stmt == new_stmt:
                            continue
                        multi_lines = new_stmt.splitlines(keepends=True)
                        if len(multi_lines) > 1:
                            for index in range(1, len(multi_lines)):
                                line_content += lines[replace_line_number + index]
                        column = update_position.file_line_column + replace_diff - 1
                        line_content = (
                            line_content[:column]
                            + new_stmt
                            + line_content[column + len(stmt) :]
                        )
                        new_line_parts = line_content.splitlines(keepends=True)
                        lines[
                            replace_line_number : replace_line_number
                            + len(new_line_parts)
                        ] = new_line_parts
                        replace_diff = replace_diff + len(new_stmt) - len(stmt)
                        LOG(
                            "info",
                            f"replace {update_position.expression_type} {update_position.file_line}:{update_position.file_line_column} [[{stmt}]] -> [[{new_stmt}]] [[{line_content}]]",
                        )
                    os.makedirs(os.path.dirname(output_source_file), exist_ok=True)
                    with open(output_source_file, "w") as output_source:
                        output_source.writelines(lines)

    def copy_source_files(self):
        if os.path.exists(self.output_directory):
            LOG("error", f"错误: 输出目录 '{self.output_directory}' 已存在")
            print(f"错误: 输出目录 '{self.output_directory}' 已存在", file=sys.stderr)
            exit(1)
        LOG("info", f"cp -r {self.input_directory} {self.output_directory}")
        result = os.system(f"cp -r {self.input_directory} {self.output_directory}")
        if result != 0:
            LOG(
                "error",
                f"错误: 复制目录时出错，返回值: {result} {self.input_directory} {self.output_directory}",
            )
            print(
                f"错误: 复制目录时出错，返回值: {result} {self.input_directory} {self.output_directory}",
                file=sys.stderr,
            )
            exit(1)

    def rename_updated_files(self):
        for old_path, new_path in self.cantidate_rename_files.items():
            os.rename(old_path, new_path)
            LOG("info", f"更名 {old_path} 为 {new_path}")

    def process_source_files(self):
        """
        处理源码文件的整体流程。

        步骤说明：
        1. 检查是否需要复制源码文件：
          - 如果 skip_copy 为 False，则调用 copy_source_files() 方法复制源码文件。

        2. 确保日志目录存在：
          - 如果 log_directory 不存在，则创建该目录（使用 os.makedirs 并设置 exist_ok=True）。

        3. 分析源码文件：
          - 首先调用 analyze_source_files_skip() 过滤处理 'SkipNamespaceDecl' 类型的声明；
          - 接着依次调用 analyze_source_files() 方法，分析以下几种类型的声明：
            • 'NamespaceDecl'
            • 'UsingDirectiveDecl'
            • 'FunctionDecl'
            • 'CallExpr'
          - 对于 'NamedDeclMemberExpr' 的分析，调用 analyze_source_files() 并附加类型标识 '*'；
          - 对于 'DeclRefExprTypeLoc' 的分析，调用 analyze_source_files() 并附加类型标识 '+'；
          - 再调用 analyze_source_files_skip() 方法过滤处理 'SkipMacroDef'；
          - 最后调用 analyze_source_files_MacroDefExpIfndefInclusionCommentSkip() 分析宏定义的相关排除条件。

        4. 更新源码文件：
          - 通过调用 replace_in_source_files() 方法在源码中替换指定内容，
          - 最后调用 rename_updated_files() 方法重命名更新后的文件。

        该方法通过上述多个步骤系统性地处理、分析并更新源码文件，确保各个阶段的处理结果能被正确利用和复用。
        """
        skip_copy = False
        if not skip_copy:
            self.copy_source_files()
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory, exist_ok=True)
        reuse_analyzed_result = False
        self.analyze_source_files_skip(
            "SkipNamespaceDecl", reuse_analyzed_result=reuse_analyzed_result
        )
        self.analyze_source_files(
            "NamespaceDecl", reuse_analyzed_result=reuse_analyzed_result
        )
        self.analyze_source_files(
            "UsingDirectiveDecl", reuse_analyzed_result=reuse_analyzed_result
        )
        self.analyze_source_files(
            "FunctionDecl", reuse_analyzed_result=reuse_analyzed_result
        )
        self.analyze_source_files(
            "CallExpr", reuse_analyzed_result=reuse_analyzed_result
        )
        self.analyze_source_files(
            "NamedDeclMemberExpr",
            reuse_analyzed_result=reuse_analyzed_result,
            append_type="*",
        )
        self.analyze_source_files(
            "DeclRefExprTypeLoc",
            reuse_analyzed_result=reuse_analyzed_result,
            append_type="+",
        )
        self.analyze_source_files_skip(
            "SkipMacroDef", reuse_analyzed_result=reuse_analyzed_result
        )
        self.analyze_source_files_MacroDefExpIfndefInclusionCommentSkip(
            reuse_analyzed_result=reuse_analyzed_result
        )
        self.replace_in_source_files()
        self.rename_updated_files()


def setup(input_directory, output_directory):
    # 构造命令：删除 input_directory/code_update 目录（如果存在），创建该目录，
    # 并进入该目录后使用 cmake 生成编译命令（compile_commands.json）
    cmd = f"rm -fr {input_directory}/code_update && mkdir -p {input_directory}/code_update && cd {input_directory}/code_update && cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    result = os.system(cmd)
    if result != 0:
        # 如果命令执行失败，打印错误信息并退出程序
        print(f"错误: 返回值: {result} 执行 {cmd}", file=sys.stderr)
        exit(1)
    # 构造命令：删除输出目录，确保输出目录不存在，为复制源文件做准备
    cmd = f"rm -fr {output_directory}"
    result = os.system(cmd)
    if result != 0:
        # 如果删除输出目录失败，打印错误信息并退出程序
        print(f"错误: 返回值: {result} 执行 {cmd}", file=sys.stderr)
        exit(1)


def complete(output_directory):
    for rm_file in RM_FILES:
        os.system(
            f"find {output_directory} -name {rm_file} -type f|xargs -i rm -fr {{}}"
        )


# 如果当前脚本作为主程序执行
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加参数：输入目录，存放源代码文件
    parser.add_argument("input_directory", type=str, help="源代码文件所在目录")
    # 添加参数：输出目录，用于保存更新后的源代码文件
    parser.add_argument(
        "output_directory", type=str, help="保存更新后的源代码文件的目录"
    )
    # 添加参数：日志目录，用于保存源代码分析日志
    parser.add_argument("log_directory", type=str, help="保存源代码分析日志的目录")
    # 添加参数：额外命令，在更新源代码文件之前执行的shell命令
    parser.add_argument("extra_cmd", type=str, help="更新源代码文件之前执行的shell命令")
    # 当没有传递任何参数时，打印帮助信息并退出
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    # 解析命令行参数
    args = parser.parse_args()
    # 检查是否提供了input_directory参数
    if not args.input_directory:
        print("错误: 必须指定 input_directory 参数", file=sys.stderr)
        exit(1)
    # 检查是否提供了output_directory参数
    if not args.output_directory:
        print("错误: 必须指定 output_directory 参数", file=sys.stderr)
        exit(1)
    # 检查是否提供了log_directory参数
    if not args.log_directory:
        print("错误: 必须指定 log_directory 参数", file=sys.stderr)
        exit(1)
    # 如果提供了extra_cmd参数，执行该命令
    if args.extra_cmd:
        LOG("info", f"{args.extra_cmd}")
        result = os.system(f"{args.extra_cmd}")
        # 如果命令执行失败，打印错误并退出
        if result != 0:
            print(
                f"错误: 执行命令时出错，返回值: {result} {args.extra_cmd}",
                file=sys.stderr,
            )
            exit(1)
    # 执行setup操作，准备工作目录
    setup(args.input_directory, args.output_directory)
    LOG("info", f"C_INCLUDE_PATH {C_INCLUDE_PATH}")
    LOG("info", f"CPLUS_INCLUDE_PATH {CPLUS_INCLUDE_PATH}")
    # 初始化SourceCodeUpdater对象，传入编译配置文件及各目录参数
    updater = SourceCodeUpdater(
        f"{args.input_directory}/code_update/compile_commands.json",
        args.input_directory,
        args.output_directory,
        args.log_directory,
    )
    # 声明全局变量g_matcher，并初始化SourceCodeMatcher对象
    global g_matcher
    g_matcher = SourceCodeMatcher(updater)
    # 处理源代码文件：复制、分析、替换、重命名
    updater.process_source_files()
    # 完成后对输出目录中的文件进行清理操作（删除不需要的文件）
    complete(args.output_directory)
    # 打印完成信息
    print("完成")
