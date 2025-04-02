import os
import argparse
import re
import json
import logging
import sys
import threading

'''
'''

# C_INCLUDE_PATH = '/home/shenda/dev/clang/clang+llvm-17.0.6-x86_64-linux-gnu-ubuntu-22.04/lib/clang/17/include:/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11:/usr/include/c++/11/backward:/usr/local/include:/usr/include/x86_64-linux-gnu:/usr/include'
# CPLUS_INCLUDE_PATH = '/home/shenda/dev/clang/clang+llvm-17.0.6-x86_64-linux-gnu-ubuntu-22.04/lib/clang/17/include:/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11:/usr/include/c++/11/backward:/usr/local/include:/usr/include/x86_64-linux-gnu:/usr/include'
# NS_STRING_DICT = {'LbsAmap':'LbsTJMap'}
# NS_SKIP_STRING_LIST = ['AMapSDK_Common']
# NS_SKIP_STRING_DICT = {}
# CLS_STRING_DICT = {'AMap':'TJMap'}
# CLS_SKIP_STRING_LIST = []
# CLS_SKIP_STRING_DICT = {}
# MC_STRING_DICT = {'LbsAmap':'LbsTJMap', 'AMAP':'TJMAP', 'AMap':'TJMap', 'Amap':'TJmap', 'amap':'tjmap'}
# MC_SKIP_STRING_LIST = ['AMAP_LOG_LEVEL_', 'AMAPCOMMON_NAMESPACE', 'Amap_Malloc'] # macro search skip string list
# MC_SKIP_STRING_DICT = {}


C_INCLUDE_PATH="/share/dev/clang+llvm-17.0.6-x86_64-linux-gnu-ubuntu-22.04/lib/clang/17/include:/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11:/usr/include/c++/11/backward:/usr/local/include:/usr/include/x86_64-linux-gnu:/usr/include"
CPLUS_INCLUDE_PATH="/share/dev/clang+llvm-17.0.6-x86_64-linux-gnu-ubuntu-22.04/lib/clang/17/include:/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11:/usr/include/c++/11/backward:/usr/local/include:/usr/include/x86_64-linux-gnu:/usr/include"
NS_STRING_DICT = {'NM':'NEW_NM'}
NS_SKIP_STRING_LIST = []
NS_SKIP_STRING_DICT = {}
CLS_STRING_DICT = {}
CLS_SKIP_STRING_LIST = []
CLS_SKIP_STRING_DICT = {}
MC_STRING_DICT = {}
MC_SKIP_STRING_LIST = []
MC_SKIP_STRING_DICT = {}


def skip_replace_file(file_path):
  skip_list = ['/binaries/', '/usr/', '/Qt/', '/open/', '__autogen']
  for skip_string in skip_list:
    if file_path.find(skip_string) != -1:
      return True
  return False

logging_lock = threading.Lock()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='update_code.log', filemode='w')
analyzer = './build/analyzer'

class StringUtils:
  def mark_string_skip_pos(self, stmt, skip_list, mark, pos_start = -1, pos_end = -1):
    new_stmt = stmt
    skip_idx = 0
    for skip_string in skip_list:
      pattern = f'{skip_string}'
      pattern_length = len(pattern)
      pos = 0
      while pos != -1:
        pos = new_stmt.find(pattern, pos)
        if pos != -1:
          if pos_start >= 0 and pos < pos_start:
            pos += pattern_length
            continue
          if pos_end >= 0 and pos + pattern_length > pos_end:
            break
          skip_mark = f'!{mark}{skip_idx}'
          new_stmt = new_stmt[:pos] + skip_mark + new_stmt[pos + pattern_length:]
          pos += len(skip_mark)
      skip_idx += 1
    return new_stmt
  def mark_string_pos(self, stmt, match_string_dict, skip_list, mark, prefix = '', suffix = '', pos_start = -1, pos_end = -1):
    new_stmt = self.mark_string_skip_pos(stmt, skip_list, mark, pos_start, pos_end)
    found = False
    match_idx = 0
    for match_string in match_string_dict.keys():
      if found:
        break
      pattern = f'{prefix}{match_string}{suffix}'
      pattern_length = len(pattern)
      pos = 0
      match_mark = f'{prefix}{mark}{match_idx}{suffix}'
      match_mark_length = len(match_mark)
      while pos != -1:
        if found:
          break
        pos = new_stmt.find(pattern, pos)
        if pos != -1:
          if pos_start >= 0 and pos < pos_start:
            pos += 1
            continue
          if pos_end >= 0 and pos + len(pattern) > pos_end:
            break
          found = True
          new_stmt = new_stmt[:pos] + match_mark + new_stmt[pos + pattern_length:]
          pos += match_mark_length
      match_idx += 1
    return found, new_stmt
  def mark_all_string_pos(self, stmt, match_string_dict, skip_list, mark, prefix = '', suffix = '', pos_start = -1, pos_end = -1):
    new_stmt = self.mark_string_skip_pos(stmt, skip_list, mark, pos_start, pos_end)
    found = False
    match_idx = 0
    for match_string in match_string_dict.keys():
      pattern = f'{prefix}{match_string}{suffix}'
      pattern_length = len(pattern)
      pos = 0
      match_mark = f'{prefix}{mark}{match_idx}{suffix}'
      match_mark_length = len(match_mark)
      while pos != -1:
        pos = new_stmt.find(pattern, pos)
        if pos != -1:
          if pos_start >= 0 and pos < pos_start:
            pos += 1
            continue
          if pos_end >= 0 and pos + len(pattern) > pos_end:
            break
          found = True
          new_stmt = new_stmt[:pos] + match_mark + new_stmt[pos + pattern_length:]
          pos += match_mark_length
      match_idx += 1
    return found, new_stmt
  def updat_string(self, stmt, match_string_dict, skip_list, mark):
    new_stmt = stmt
    skip_idx = 0
    for value in skip_list:
      new_stmt = new_stmt.replace(f'!{mark}{skip_idx}', value)
      skip_idx += 1
    match_idx = 0
    for value in match_string_dict.values():
      new_stmt = new_stmt.replace(f'{mark}{match_idx}', value)
      match_idx += 1
    return new_stmt

def LOG(level, message):
  with logging_lock:
    if level == 'info':
      logging.info(message)
    elif level == 'warning':
      logging.warning(message)
    elif level == 'error':
      logging.error(message)

class SourceCodeUpdateInfo:
  def __init__(self, file_path):
    self.file_path = file_path
    self.update_position = {}

class SourceCodeUpdatePosition:
  def __init__(self, parent, file_path, file_line, file_line_column, log_entry_type, expression_type, stmt, tostmt=None):
    self.parent = parent
    self.file_path = file_path
    self.file_line = file_line
    self.file_line_column = file_line_column
    self.log_entry_type = log_entry_type
    self.expression_type = expression_type
    self.stmt = stmt
    self.tostmt = tostmt
  def match(self, file_path, file_line, file_line_column, stmt, expression_type, tostmt):
    if self.file_line == file_line:
      current_start = self.file_line_column
      current_length = len(self.stmt)
      current_end = self.file_line_column + current_length
      new_start = file_line_column
      new_length = len(stmt)
      new_end = file_line_column + new_length
      if new_start < current_end and new_end > current_start:
        # return True
        if new_start >= current_start and new_end <= current_end and new_length < current_length:
          pos_start = new_start - current_start
          pos_end = pos_start + new_length
          sub_stmt = self.stmt[pos_start:pos_end]
          new_sub_stmt, _ = self.parent.match_expression(file_path, file_line, pos_start, sub_stmt, self.expression_type)
          if len(new_sub_stmt) == 0:
            new_sub_stmt, _ = self.parent.match_expression(file_path, file_line, pos_start, sub_stmt, expression_type)
          if sub_stmt == stmt:
            if new_sub_stmt == tostmt:
              return True
            return None
        elif new_start <= current_start and new_end >= current_end and new_length > current_length:
          pos_start = current_start - new_start
          pos_end = pos_start + current_length
          sub_stmt = stmt[pos_start:pos_end]
          new_sub_stmt, _ = self.parent.match_expression(file_path, file_line, pos_start, sub_stmt, expression_type)
          if len(new_sub_stmt) == 0:
            new_sub_stmt, _ = self.parent.match_expression(file_path, file_line, pos_start, sub_stmt, self.expression_type)
          if sub_stmt == self.stmt:
            if new_sub_stmt == self.tostmt:
              self.stmt = self.tostmt = ''
            else:
              return None
        else:
          return True
    return False

class SourceCodeUpdater:
  def __init__(self, compile_commands_json, input_directory, output_directory, log_directory):
    self.compile_commands_json = compile_commands_json
    self.source_files = {}
    self.parse_compile_commands()
    self.input_directory = os.path.realpath(input_directory)
    self.output_directory = os.path.realpath(output_directory)
    self.log_directory = os.path.realpath(log_directory)
    if not os.path.exists(self.compile_commands_json):
      print(f"错误: 编译配置 '{self.compile_commands_json}' 不存在", file=sys.stderr)
      exit(1)
    if not os.path.exists(self.input_directory):
      print(f"错误: 输入目录 '{self.input_directory}' 不存在", file=sys.stderr)
      exit(1)
    if os.path.exists(self.output_directory):
      LOG('warning', f"输出目录 '{self.output_directory}' 已存在")
    if os.path.exists(self.log_directory):
      LOG('warning', f"日志目录 '{self.log_directory}' 已存在")
    self.cantidate_replace_files = {}
    self.cantidate_rename_files = {}
    self.lock = threading.Lock()
  
  def parse_compile_commands(self):
    with open(self.compile_commands_json, 'r') as f:
      compile_commands = json.load(f)
    for entry in compile_commands:
      command = entry['command']
      compile_commands = command.split(' -o ')[0]
      source_file = os.path.abspath(entry['file'])
      if skip_replace_file(source_file):
        return
      # process file filter #
      # if source_file != '/share/dev/mapdev/MMShell_bak/open/AMapCommon/src/overlay/renderers/POILabel/POIRenderer.cpp':
      #   continue
      self.source_files[source_file] = compile_commands
    if len(self.source_files) == 0:
      print(f"错误: 没有从{self.compile_commands_json}找到任何编译配置", file=sys.stderr)
      exit(1)

  def exec_analyzer(self, analyzer_log_file, cmd, files, times=-1, skip=True):
    if skip == True:
      return
    result = os.system(f'rm -fr {analyzer_log_file}')
    if result != 0:
      print(f"错误: 返回值: {result} 删除{analyzer_log_file}", file=sys.stderr)
      exit(1)
    count = 0
    for file_path in files:
      if times != -1 and count >= times:
        return
      count += 1
      LOG('info', f"分析{cmd}: {file_path}")
      compile_commands = self.source_files.get(file_path)
      # filter
      # if file_path != '/home/shenda/dev/mapdev/MMShell_1500/AMap3D-Cpp/MapsLibrary/Source/AMapAdapter.cpp':
      #   continue
      os.environ["C_INCLUDE_PATH"] = C_INCLUDE_PATH
      os.environ["CPLUS_INCLUDE_PATH"] = CPLUS_INCLUDE_PATH
      if compile_commands is None:
        # LOG('info', f'echo "#\n#\n# {file_path}\n#\n#" >> {analyzer_log_file} && ANALYZE_CMD="{cmd}" {analyzer} {file_path} -- {compile_commands} -Wno-everything >> {analyzer_log_file}')
        result = os.system(f'echo "#\n#\n# {file_path}\n#\n#" >> {analyzer_log_file} && ANALYZE_CMD="{cmd}" {analyzer} {file_path} -p {os.path.dirname(self.compile_commands_json)} >> {analyzer_log_file}')
      else:
        # LOG('info', f'echo "#\n#\n# {file_path}\n#\n#" >> {analyzer_log_file} && ANALYZE_CMD="{cmd}" {analyzer} {file_path} -- {compile_commands} -Wno-everything >> {analyzer_log_file}')
        result = os.system(f'echo "#\n#\n# {file_path}\n#\n#" >> {analyzer_log_file} && ANALYZE_CMD="{cmd}" {analyzer} {file_path} -- {compile_commands} -Wno-everything >> {analyzer_log_file}')
      if result != 0:
        print(f"错误: 分析源文件时出错，返回值: {result} {file_path}", file=sys.stderr)
        exit(1)

  def match_update_position(self, file_path, file_line, file_line_column, expression_type, stmt, tostmt):
    with self.lock:
      if file_path not in self.cantidate_replace_files:
        self.cantidate_replace_files[file_path] = SourceCodeUpdateInfo(file_path)
    def match_stmt_at_position(file_path, file_line, file_line_column, stmt):
      with open(file_path, 'r') as f:
          lines = f.readlines()
      if file_line < 1 or file_line > len(lines):
          return f'bad file line {file_path}:{file_line}'
      line = lines[file_line - 1]
      col_index = file_line_column - 1
      if col_index < 0 or col_index > len(line):
          return f'bad file line column {file_path}:{file_line}:{file_line_column}'
      file_stmt = ''
      multi_lines = stmt.splitlines(keepends = True)
      if len(multi_lines) > 1:
        file_stmt += line[col_index:]
        index = 1
        for _ in range(1, len(multi_lines) - 1):
          file_stmt += lines[file_line - 1 + index]
          index += 1
        file_stmt += lines[file_line - 1 + index][:len(multi_lines[-1])]
      else:
        file_stmt = line[col_index:col_index + len(multi_lines[-1])]
      if file_stmt != stmt:
        return f'content not match {file_path}:{file_line}:{file_line_column} [[{stmt}]] != [[{file_stmt}]]'
      return None
    match_error = match_stmt_at_position(file_path, file_line, file_line_column, stmt)
    if match_error:
      LOG('warning', f'忽略不匹配的源码内容 {expression_type} {match_error}')
      return True
    with self.lock:
      for pos in self.cantidate_replace_files[file_path].update_position.values():
        result = pos.match(file_path, file_line, file_line_column, stmt, expression_type, tostmt)
        if result is None:
          LOG('error', f'错误替换 {expression_type} {file_path}:{file_line}:{file_line_column} [[{stmt}]] -> [[{tostmt}]]')
          LOG('error', f'已存在替换 {pos.log_entry_type} {pos.expression_type} {pos.line}:{pos.column} [[{pos.stmt}]] -> [[{pos.tostmt}]]')
          return True
        elif result:
          return True
    return False

  def match_macro(self, file_path, file_line, file_line_column, macroname, macrotype, macrostmt):
    str_utils = StringUtils()
    if macrotype == 'Ifndef':
      new_stmt = macrostmt
      found_mc, new_stmt = str_utils.mark_all_string_pos(new_stmt, MC_STRING_DICT, [], '_MC_')
      if found_mc:
        new_stmt = str_utils.updat_string(new_stmt, MC_STRING_DICT, [], '_MC_')
        return new_stmt
    elif macrotype == 'MacroExpands':
      new_stmt = macrostmt
      pos = new_stmt.find('(')
      skip_list = MC_SKIP_STRING_LIST
      found_mc, new_stmt = str_utils.mark_all_string_pos(new_stmt, MC_STRING_DICT, skip_list, '_MC_', '', '', -1, pos)
      if found_mc:
        new_stmt = str_utils.updat_string(new_stmt, MC_STRING_DICT, skip_list, '_MC_')
        return new_stmt
    elif macrotype == 'MacroDefined':
      new_stmt = macrostmt
      skip_list = MC_SKIP_STRING_LIST
      found_mc, new_stmt = str_utils.mark_all_string_pos(new_stmt, MC_STRING_DICT, skip_list, '_MC_')
      if found_mc:
        new_stmt = str_utils.updat_string(new_stmt, MC_STRING_DICT, skip_list, '_MC_')
        return new_stmt
    return None

  def match_inclusion(self, file_path, macrostmt):
    # return None
    if skip_replace_file(file_path):
      return None
    str_utils = StringUtils()
    new_stmt = macrostmt
    pos = macrostmt.rfind('/')
    found_mc, new_stmt = str_utils.mark_all_string_pos(new_stmt, MC_STRING_DICT, [], '_INC_', '', '', pos)
    if found_mc:
      new_stmt = str_utils.updat_string(new_stmt, MC_STRING_DICT, [], '_INC_')
      real_file_path = os.path.join(self.output_directory, os.path.relpath(file_path, self.input_directory))
      if real_file_path not in self.cantidate_rename_files:
        dir_name = os.path.dirname(real_file_path)
        file_name = os.path.basename(real_file_path)
        _, file_name = str_utils.mark_all_string_pos(file_name, MC_STRING_DICT, [], '_INC_')
        file_name = str_utils.updat_string(file_name, MC_STRING_DICT, [], '_INC_')
        self.cantidate_rename_files[real_file_path] = os.path.join(dir_name, file_name)
      return new_stmt
    return None

  def analyze_source_files_MacroDefExp(self, reuse_analyzed_result = False):
    analyze_log_file = f'{self.log_directory}/analyze_MacroDefExp.log'
    self.exec_analyzer(analyze_log_file, "MacroDefExp|", self.source_files.keys(), -1, reuse_analyzed_result)
    with open(analyze_log_file, 'r') as analyze_log_file_content:
      for line in analyze_log_file_content:
        if line.startswith('#'):
          continue
        try:
          log_entry = json.loads(line)
          macrotype = log_entry['type']
          if macrotype == 'MacroExpands' or macrotype == 'MacroDefined' or macrotype == 'Ifndef':
            file_path = log_entry['file']
            file_line = log_entry['line']
            file_line_column = log_entry['column']
            macroname = log_entry['macroname']
            macrostmt = log_entry['macrostmt']
            new_macrostmt = self.match_macro(file_path, file_line, file_line_column, macroname, macrotype, macrostmt)
            if new_macrostmt is not None:
              if not self.match_update_position(file_path, file_line, file_line_column, macrotype, macrostmt, new_macrostmt):
                with self.lock:
                  self.cantidate_replace_files[file_path].update_position[(file_line, file_line_column)] = SourceCodeUpdatePosition(self, file_path, file_line, file_line_column, 'macro', macrotype, macrostmt, new_macrostmt)
          elif macrotype == 'InclusionDirective':
            file_path = log_entry['file']
            file_line = log_entry['line']
            file_column = log_entry['column']
            include_file_path = log_entry['includefilepath']
            stmt = log_entry['stmt']
            new_stmt = self.match_inclusion(include_file_path, stmt)
            if new_stmt is not None:
              if not self.match_update_position(file_path, file_line, file_column, macrotype, stmt, new_stmt):
                with self.lock:
                  self.cantidate_replace_files[file_path].update_position[(file_line, file_column)] = SourceCodeUpdatePosition(self, file_path, file_line, file_column, 'macro', macrotype, stmt, new_stmt)
        except json.JSONDecodeError as e:
          print(f"错误: 解析宏分析日志时出错: {e} {line}", file=sys.stderr)
          exit(1)

  def match_expression(self, file_path, file_line, file_line_column, stmt, expression_type, dfile_path = None):
    if len(stmt) == 0:
      return '', ''
    str_utils = StringUtils()
    NS_SKIP = NS_SKIP_STRING_LIST + list(NS_SKIP_STRING_DICT.keys())
    CLS_SKIP = CLS_SKIP_STRING_LIST + list(CLS_SKIP_STRING_DICT.keys())
    if expression_type.endswith('+'):
      pos = stmt.find('<')
      if pos != -1:
        stmt = stmt[:stmt.find('<')]
      new_stmt = stmt
      found_ns, new_stmt = str_utils.mark_all_string_pos(new_stmt, NS_STRING_DICT, NS_SKIP, '_NS_', '', '::')
      found_cls, new_stmt = str_utils.mark_all_string_pos(new_stmt, CLS_STRING_DICT, CLS_SKIP, '_CLS_')
      if found_ns or found_cls:
        new_stmt = str_utils.updat_string(new_stmt, NS_STRING_DICT, NS_SKIP, '_NS_')
        new_stmt = str_utils.updat_string(new_stmt, CLS_STRING_DICT, CLS_SKIP, '_CLS_')
        return new_stmt, stmt
    elif expression_type.endswith('*'):
      found, new_stmt = str_utils.mark_string_pos(stmt, CLS_STRING_DICT, CLS_SKIP, '_CLS_')
      if found:
        new_stmt = str_utils.updat_string(new_stmt, CLS_STRING_DICT, CLS_SKIP, '_CLS_')
        return new_stmt, stmt
    elif expression_type == 'NamespaceDecl':
      found, new_stmt = str_utils.mark_string_pos(stmt, NS_STRING_DICT, NS_SKIP, '_NS_')
      if found:
        new_stmt = str_utils.updat_string(new_stmt, NS_STRING_DICT, NS_SKIP, '_NS_')
        return new_stmt, stmt
    elif expression_type == 'NamedDecl' or expression_type == 'MemberExpr':
      found, new_stmt = str_utils.mark_string_pos(stmt, CLS_STRING_DICT, CLS_SKIP, '_CLS_')
      if found:
        new_stmt = str_utils.updat_string(new_stmt, CLS_STRING_DICT, CLS_SKIP, '_CLS_')
        return new_stmt, stmt
    elif expression_type == 'CallExpr':
      stmt_list = stmt.split('%%')
      if len(stmt_list) != 2:
        return '', ''
      if len(stmt_list[0]) == 0:
        return '', ''
      func_pos_start = stmt_list[0].find(stmt_list[1])
      if func_pos_start == -1:
        return '', ''
      func_pos_end = func_pos_start + len(stmt_list[1])
      stmt_list[0] = stmt_list[0][func_pos_start:func_pos_end]
      found, new_stmt = str_utils.mark_string_pos(stmt_list[0], CLS_STRING_DICT, CLS_SKIP, '_CLS_')
      if found:
        new_stmt = str_utils.updat_string(new_stmt, CLS_STRING_DICT, CLS_SKIP, '_CLS_')
        return new_stmt, stmt_list[0]
    elif expression_type == 'FunctionDecl' or expression_type == 'FunctionImpl':
      function_exp_list = stmt.split('%%')
      function_prefix = ''
      if len(function_exp_list) == 2:
        pos_prefix = stmt.find(function_exp_list[1])
        if pos_prefix != -1:
          function_prefix = stmt[:pos_prefix + len(function_exp_list[1])]
          stmt = stmt[len(function_prefix):]
      pos_call = stmt.find('(')
      if pos_call != -1:
        stmt = stmt[:pos_call]
      new_stmt = stmt
      found_ns, new_stmt = str_utils.mark_all_string_pos(new_stmt, NS_STRING_DICT, NS_SKIP, '_NS_', '', '::')
      found_cls, new_stmt = str_utils.mark_all_string_pos(new_stmt, CLS_STRING_DICT, CLS_SKIP, '_CLS_')
      if found_ns or found_cls:
        new_stmt = str_utils.updat_string(new_stmt, NS_STRING_DICT, NS_SKIP, '_NS_')
        new_stmt = str_utils.updat_string(new_stmt, CLS_STRING_DICT, CLS_SKIP, '_CLS_')
        return new_stmt, stmt
    elif expression_type == 'UsingDirectiveDecl':
      namespace_pattern = r'using\s+namespace\s+'
      namespace_pos = re.search(namespace_pattern, stmt)
      found_ns = False
      found_cls = False
      if namespace_pos:
        new_stmt = stmt
        found_ns, new_stmt = str_utils.mark_all_string_pos(new_stmt, NS_STRING_DICT, NS_SKIP, '_NS_')
        if found_ns:
          new_stmt = str_utils.updat_string(new_stmt, NS_STRING_DICT, NS_SKIP, '_NS_')
          return new_stmt, stmt
      else:
        found_ns, new_stmt = str_utils.mark_all_string_pos(new_stmt, NS_STRING_DICT, NS_SKIP, '_NS_', '', '::')
        found_cls, new_stmt = str_utils.mark_all_string_pos(new_stmt, CLS_STRING_DICT, CLS_SKIP, '_CLS_')
        if found_ns or found_cls:
          new_stmt = str_utils.updat_string(new_stmt, NS_STRING_DICT, NS_SKIP, '_NS_')
          new_stmt = str_utils.updat_string(new_stmt, CLS_STRING_DICT, CLS_SKIP, '_CLS_')
          return new_stmt, stmt
    else:
      LOG('error', f'unchecked expression type: {expression_type}')
    return '', ''

  def check_log_entry(self, log_entry, log_entry_type, append_type = ''):
    if log_entry['type'] != log_entry_type:
      return
    file_path = log_entry['file']
    if len(file_path) == 0:
      return
    if skip_replace_file(file_path):
      return
    file_line = log_entry['line']
    file_line_column = log_entry['column']
    stmt = log_entry['stmt']
    expression_type = log_entry['exptype']
    dfile_path = log_entry['dfile']
    new_stmt, adj_stmt = self.match_expression(file_path, file_line, file_line_column, stmt, expression_type + append_type, dfile_path)
    if len(new_stmt) > 0:
      multi_lines = False
      while adj_stmt[0] in ['\n', '\r']:
        multi_lines = True
        adj_stmt = adj_stmt.lstrip('\r\n')
        new_stmt = new_stmt.lstrip('\r\n')
        file_line += 1
        file_line_column = 1
      if not multi_lines:
        pos = stmt.find(adj_stmt)
        if pos == -1:
          LOG('error', f'表达式匹配错误 {log_entry["file"]} {log_entry["line"]} {log_entry["column"]} {log_entry["stmt"]} {log_entry["exptype"]}')
          return
        file_line_column += pos
      stmt = adj_stmt
      if not self.match_update_position(file_path, file_line, file_line_column, expression_type + append_type, stmt, new_stmt):
        with self.lock:
          self.cantidate_replace_files[file_path].update_position[(file_line, file_line_column)] = SourceCodeUpdatePosition(self, file_path, file_line, file_line_column, log_entry_type, expression_type + append_type, stmt, new_stmt)

  def analyze_source_files(self, analyze_type, append_type = '', reuse_analyzed_result = True):
    analyze_log_file = f'{self.log_directory}/analyze_{analyze_type}.log'
    self.exec_analyzer(analyze_log_file, f'{analyze_type}|', self.source_files.keys(), -1, reuse_analyzed_result)
    with open(analyze_log_file, 'r') as analyze_log_file_content:
      line_no = 0
      for line in analyze_log_file_content:
        line_no += 1
        if line.startswith('#'):
          continue
        try:
          log_entry = json.loads(line)
          self.check_log_entry(log_entry, analyze_type, append_type)
          # return
        except json.JSONDecodeError as e:
          print(f'错误: 解析{analyze_log_file} {line_no}行出错: {e} {line}', file=sys.stderr)
          exit(1)
    return

  def replace_in_source_files(self):
    for source_file, update_info in self.cantidate_replace_files.items():
      if skip_replace_file(source_file):
        continue
      output_source_file = os.path.join(self.output_directory, os.path.relpath(source_file, self.input_directory))
      LOG('info', f"from file: {source_file}")
      LOG('info', f"to file: {output_source_file}")
      os.system(f'cp {source_file} {output_source_file}')
      lines = []
      with open(source_file, 'r') as input_source:
        try:
          lines = input_source.readlines()
        except Exception as e:
          print(f"Error reading file {source_file}: {e}")
          continue
        if len(update_info.update_position) > 0:
          sorted_positions = sorted(update_info.update_position.values(), key=lambda pos: (pos.file_line, pos.file_line_column))
          replace_line_number = -1
          replace_diff = 0
          line_content = ''
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
            line_content = line_content[:column] + new_stmt + line_content[column + len(stmt):]
            new_line_parts = line_content.splitlines(keepends=True)
            lines[replace_line_number:replace_line_number + len(new_line_parts)] = new_line_parts
            replace_diff = replace_diff + len(new_stmt) - len(stmt)
            LOG('info', f'replace {update_position.log_entry_type}:{update_position.expression_type} {update_position.file_line}:{update_position.file_line_column} [[{stmt}]] -> [[{new_stmt}]] [[{line_content}]]')
          os.makedirs(os.path.dirname(output_source_file), exist_ok=True)
          with open(output_source_file, 'w') as output_source:
            output_source.writelines(lines)

  def copy_source_files(self):
    if os.path.exists(self.output_directory):
      print(f"错误: 输出目录 '{self.output_directory}' 已存在", file=sys.stderr)
      exit(1)
    LOG('info', f'cp -r {self.input_directory} {self.output_directory}')
    result = os.system(f'cp -r {self.input_directory} {self.output_directory}')
    if result != 0:
      print(f"错误: 复制目录时出错，返回值: {result} {self.input_directory} {self.output_directory}", file=sys.stderr)
      exit(1)

  def rename_updated_files(self):
    for old_path, new_path in self.cantidate_rename_files.items():
      os.rename(old_path, new_path)
      LOG('info', f"更名 {old_path} 为 {new_path}")

  def process_source_files(self):
    skip_copy = False
    # skip_copy = True
    if not skip_copy:
      self.copy_source_files()
    if not os.path.exists(self.log_directory):
      os.makedirs(self.log_directory, exist_ok=True)
    reuse_analyzed_result = False
    # reuse_analyzed_result = True
    threads = []
    analyze_types = [
      ('NamespaceDecl', ''),
      ('UsingDirectiveDecl', ''),
      ('FunctionDecl', ''),
      ('CallExpr', ''),
      ('NamedDeclMemberExpr', '*'),
      ('DeclRefExprTypeLoc', '+'),
    ]
    if False:
      for analyze_type, append_type in analyze_types:
        thread = threading.Thread(target=self.analyze_source_files, args=(analyze_type,), kwargs={'append_type': append_type, 'reuse_analyzed_result': reuse_analyzed_result},)
        threads.append(thread)
        thread.start()
      thread = threading.Thread(target=self.analyze_source_files_MacroDefExp, args=(reuse_analyzed_result,))
      threads.append(thread)
      thread.start()
      for thread in threads:
        thread.join()
    else:
      self.analyze_source_files('NamespaceDecl', reuse_analyzed_result=reuse_analyzed_result)
      self.analyze_source_files('UsingDirectiveDecl', reuse_analyzed_result=reuse_analyzed_result)
      self.analyze_source_files('FunctionDecl', reuse_analyzed_result=reuse_analyzed_result)
      self.analyze_source_files('CallExpr', reuse_analyzed_result=reuse_analyzed_result)
      self.analyze_source_files('NamedDeclMemberExpr', reuse_analyzed_result=reuse_analyzed_result, append_type='*')
      self.analyze_source_files('DeclRefExprTypeLoc', reuse_analyzed_result=reuse_analyzed_result, append_type='+')
      self.analyze_source_files_MacroDefExp(reuse_analyzed_result=reuse_analyzed_result)
    self.replace_in_source_files()
    self.rename_updated_files()

# def clang_check():
#   os.environ["C_INCLUDE_PATH"] = C_INCLUDE_PATH
#   os.environ["CPLUS_INCLUDE_PATH"] = CPLUS_INCLUDE_PATH
#   result = os.system(f'clang-check -p /home/shenda/dev/mapdev/MMShell_1500/build /home/shenda/dev/mapdev/MMShell_1500/NavigationSDK-Cpp/NavigationLibrary/Source/CarOverlay.cpp --ast-dump --ast-dump-filter=AMap')

if __name__ == "__main__":
  # clang_check()
  # exit(0)
  parser = argparse.ArgumentParser()
  parser.add_argument('compile_commands_json', type=str, help='Path to compile_commands.json')
  parser.add_argument('input_directory', type=str, help='The directory of source code files')
  parser.add_argument('output_directory', type=str, help='The directory to save the updated source code files')
  parser.add_argument('log_directory', type=str, help='The directory to save source code analysis files')
  parser.add_argument('extra_cmd', type=str, help='The shell command to execute before updating source code files')
  if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)
  args = parser.parse_args()
  if not args.compile_commands_json:
    print("错误: 必须指定 compile_commands_json 参数", file=sys.stderr)
    exit(1)
  if not args.input_directory:
    print("错误: 必须指定 input_directory 参数", file=sys.stderr)
    exit(1)
  if not args.output_directory:
    print("错误: 必须指定 output_directory 参数", file=sys.stderr)
    exit(1)
  if not args.log_directory:
    print("错误: 必须指定 log_directory 参数", file=sys.stderr)
    exit(1)
  if args.extra_cmd:
    LOG('info', f'{args.extra_cmd}')
    result = os.system(f'{args.extra_cmd}')
    if result != 0:
      print(f"错误: 执行命令时出错，返回值: {result} {args.extra_cmd}", file=sys.stderr)
      exit(1)

  updater = SourceCodeUpdater(args.compile_commands_json, args.input_directory, args.output_directory, args.log_directory)
  updater.process_source_files()
  print("完成")
