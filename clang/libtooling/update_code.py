import os
import argparse
import re
import json
import logging
import sys
import threading

'''
'''

AMAP_NS_STRING_LIST = ['NM']
AMAP_NS_NEW_STRING = 'NewNM'
AMAP_CLS_STRING_LIST = ['ns', 'NM', 'NSDEF']
AMAP_CLS_NEW_STRING = 'NewNM'

# AMAP_NS_STRING_LIST = ['LbsAmap', 'NS']
# AMAP_NS_NEW_STRING = 'LbsTJ'
# AMAP_CLS_STRING_LIST = ['AMap']
# AMAP_CLS_NEW_STRING = 'TJ'

def skip_replace_file(file_path):
  skip_list = ['/binaries/', '/usr/', '/Qt/', '/open/', '/output/', '__autogen']
  for skip_string in skip_list:
    if file_path.find(skip_string) != -1:
      return True
  return False

logging_lock = threading.Lock()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='update_code.log', filemode='w')
analyzer = './build/analyzer'

def mark_string_pos(stmt, match_string_list, mark, prefix = '', suffix = '', pos_start = -1, pos_end = -1):
  new_stmt = stmt
  found = False
  for match_string in match_string_list:
    if found:
      break
    pattern = f'{prefix}{match_string}{suffix}'
    pos = 0
    while pos != -1:
      if found:
        break
      pos = new_stmt.find(pattern, pos)
      if pos != -1:
        if pos_start >= 0 and pos < pos_start:
          pos += 1
          continue
        if pos_end >= 0 and pos > pos_end:
          pos += 1
          continue
        found = True
        new_stmt = new_stmt[:pos] + mark + new_stmt[pos + len(pattern):]
  return found, new_stmt
def mark_all_string_pos(stmt, match_string_list, mark, prefix = '', suffix = '', pos_start = -1, pos_end = -1):
  new_stmt = stmt
  found = False
  for match_string in match_string_list:
    pattern = f'{prefix}{match_string}{suffix}'
    pos = 0
    while pos != -1:
      pos = new_stmt.find(pattern, pos)
      if pos != -1:
        if pos_start >= 0 and pos < pos_start:
          pos += 1
          continue
        if pos_end >= 0 and pos > pos_end:
          pos += 1
          continue
        found = True
        new_stmt = new_stmt[:pos] + mark + new_stmt[pos + len(pattern):]
  return found, new_stmt

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
  def __init__(self, line, column, kind, stmt, tostmt=None, type='expression'):
    self.line = line
    self.column = column
    self.kind = kind
    self.stmt = stmt
    self.tostmt = tostmt
    self.type = type
  def match(self, line, column, stmt, kind):
    if self.line == line:
      current_start = self.column
      current_end = self.column + len(self.stmt)
      new_start = column
      new_length = len(stmt)
      new_end = column + new_length
      if new_start < current_end and new_end > current_start:
        diff = current_end - new_start
        if new_start > current_start and diff > 0 and new_length < len(self.stmt):
          pos_start = new_start - current_start
          pos_end = pos_start + len(stmt)
          if self.stmt[pos_start:pos_end] == stmt:
            self.stmt = self.stmt[:len(self.stmt) - diff]
            self.tostmt = self.stmt
            found_ns, self.tostmt = mark_all_string_pos(self.tostmt, AMAP_NS_STRING_LIST, '|+|')
            found_cls, self.tostmt = mark_all_string_pos(self.tostmt, AMAP_CLS_STRING_LIST, '|-|')
            if found_ns or found_cls:
              self.tostmt = self.tostmt.replace('|+|', AMAP_NS_NEW_STRING)
              self.tostmt = self.tostmt.replace('|-|', AMAP_CLS_NEW_STRING)
            return False
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
    self.rename_file_list = []
    self.source_files_dependencies = {}

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
      if compile_commands is None:
        LOG('info', f'echo "#\n#\n# {file_path}\n#\n#" >> {analyzer_log_file} && ANALYZE_CMD="{cmd}" {analyzer} {file_path} -- {compile_commands} -Wno-everything >> {analyzer_log_file}')
        result = os.system(f'echo "#\n#\n# {file_path}\n#\n#" >> {analyzer_log_file} && ANALYZE_CMD="{cmd}" {analyzer} {file_path} -p {os.path.dirname(self.compile_commands_json)} >> {analyzer_log_file}')
      else:
        LOG('info', f'echo "#\n#\n# {file_path}\n#\n#" >> {analyzer_log_file} && ANALYZE_CMD="{cmd}" {analyzer} {file_path} -- {compile_commands} -Wno-everything >> {analyzer_log_file}')
        result = os.system(f'echo "#\n#\n# {file_path}\n#\n#" >> {analyzer_log_file} && ANALYZE_CMD="{cmd}" {analyzer} {file_path} -- {compile_commands} -Wno-everything >> {analyzer_log_file}')
      if result != 0:
        print(f"错误: 分析源文件时出错，返回值: {result} {file_path}", file=sys.stderr)
        exit(1)

  def match_update_position(self, file_path, line, column, stmt, expression_type):
    if file_path not in self.cantidate_replace_files:
      self.cantidate_replace_files[file_path] = SourceCodeUpdateInfo(file_path)
    for pos in self.cantidate_replace_files[file_path].update_position.values():
      if pos.match(line, column, stmt, expression_type):
        return True
    return False

  # def analyze_source_files_InclusionDirective(self):
  #   analyzer_log_file = 'analyzer_InclusionDirective.log'
  #   self.exec_analyzer(analyzer_log_file, "InclusionDirective|", self.source_files.keys(), -1, reuse_analyzed_result)
  #   with open(analyzer_log_file, 'r') as analyzer_log_file_content:
  #     for line in analyzer_log_file_content:
  #       if line.startswith('#'):
  #         continue
  #       try:
  #         log_entry = json.loads(line)
  #         if log_entry['type'] != 'InclusionDirective':
  #           continue
  #         filepath = log_entry['file']
  #         includefilepath = log_entry['includefilepath']
  #         if filepath == includefilepath:
  #           continue
  #         if self.source_files_dependencies.get(includefilepath) is None:
  #           self.source_files_dependencies[includefilepath] = []
  #         if filepath not in self.source_files_dependencies[includefilepath]:
  #           self.source_files_dependencies[includefilepath].append(filepath)
  #       except json.JSONDecodeError as e:
  #         print(f"错误: 解析include分析日志时出错: {e} {line}", file=sys.stderr)
  #         exit(1)

  def match_macro(self, macroname, macrotype, macrostmt):
    if macrotype == 'MacroExpands':
      # if macroname == 'BEGIN_NS':
      #   return f'namespace {AMAP_NS_NEW_STRING} {{'
      # elif macroname == 'END_NS':
      #   return f'}}'
      return None
    elif macrotype == 'MacroDefined':
      found = False
      def_pos = macrostmt.find(macroname)
      if def_pos == -1:
        return ''
      def_pos += len(macroname)
      new_stmt = macrostmt
      found_ns, new_stmt = mark_all_string_pos(new_stmt, AMAP_NS_STRING_LIST, '|+|', '', '', def_pos)
      found_cls, new_stmt = mark_all_string_pos(new_stmt, AMAP_CLS_STRING_LIST, '|-|', '', '', def_pos)
      if found_ns or found_cls:
        new_stmt = new_stmt.replace('|+|', AMAP_NS_NEW_STRING)
        new_stmt = new_stmt.replace('|-|', AMAP_CLS_NEW_STRING)
        return new_stmt
      return ''
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
          if macrotype == 'MacroExpands' or macrotype == 'MacroDefined':
            file_path = log_entry['file']
            file_line = log_entry['line']
            file_column = log_entry['column']
            macroname = log_entry['macroname']
            macrostmt = log_entry['macrostmt']
            matchupdate = self.match_macro(macroname, macrotype, macrostmt)
            if matchupdate is not None and len(matchupdate) > 0:
              if not self.match_update_position(file_path, file_line, file_column, macrostmt, macrotype):
                self.cantidate_replace_files[file_path].update_position[(file_line, file_column)] = SourceCodeUpdatePosition(file_line, file_column, 'MacroDefExp', macrostmt, matchupdate)
        except json.JSONDecodeError as e:
          print(f"错误: 解析include分析日志时出错: {e} {line}", file=sys.stderr)
          exit(1)

  def skip_stmt(self, stmt, expression_type):
    for ns in ['AMapSDK_Common::']:
      if  ns in stmt:
        return True
    # if expression_type == 'Elaborated+':
    #   pattern_list = ['AMapNaviCoreEyrieViewWrap', 'AMapNaviCoreEyrieObserverImpl', 'AMapNaviCoreEyrieViewManager', 'AMapNaviCoreManager', 'AMapNaviCoreObserver']
    #   if "AMapNaviCore" in stmt:
    #     if not any(pattern in stmt for pattern in pattern_list):
    #       return True
    #   pattern_list = ['AMapNaviRouteNotifyDataType', 'AMapNaviRouteGuideGroup', 'AMapNaviRouteGuideSegment', 'AMapNaviRouteNotifyData']
    #   if "AMapNaviRoute" in stmt:
    #     if not any(pattern in stmt for pattern in pattern_list):
    #       return True
    # pattern_list = ['DIMENSION_NUM']
    # if "DIME" in stmt:
    #   if any(pattern in stmt for pattern in pattern_list):
    #     return True
    return False

  def match_expression(self, stmt, expression_type, dfile_path = None):
    if len(stmt) == 0:
      return '', ''
    if expression_type.endswith('+'):
      stmt = re.sub(r'\bconst\s+', '', stmt)
      if '<' in stmt:
        stmt = stmt[:stmt.find('<')]
      new_stmt = stmt
      if expression_type == 'Elaborated+':
        if self.skip_stmt(new_stmt, expression_type):
          return '', ''
        found_ns, new_stmt = mark_all_string_pos(new_stmt, AMAP_NS_STRING_LIST, '|+|', '', '::')
        found_cls, new_stmt = mark_all_string_pos(new_stmt, AMAP_CLS_STRING_LIST, '|-|')
        if found_ns or found_cls:
          new_stmt = new_stmt.replace('|+|', f'{AMAP_NS_NEW_STRING}::')
          new_stmt = new_stmt.replace('|-|', AMAP_CLS_NEW_STRING)
          return new_stmt, stmt
        return '', ''
      else:
        found_cls, new_stmt = mark_string_pos(new_stmt, AMAP_CLS_STRING_LIST, '|-|')
        if found_cls:
          new_stmt = new_stmt.replace('|-|', AMAP_CLS_NEW_STRING)
          return new_stmt, stmt
        return '', ''
    elif expression_type.endswith('='):
      if expression_type == 'FunctionProto=':
        if skip_replace_file(dfile_path):
          return '', ''
      new_stmt = stmt
      found_ns, new_stmt = mark_all_string_pos(new_stmt, AMAP_NS_STRING_LIST, '|+|', '', '::')
      found_cls, new_stmt = mark_all_string_pos(new_stmt, AMAP_CLS_STRING_LIST, '|-|', '', '::')
      if found_ns or found_cls:
        new_stmt = new_stmt.replace('|+|', f'{AMAP_NS_NEW_STRING}::')
        new_stmt = new_stmt.replace('|-|', f'{AMAP_CLS_NEW_STRING}::')
        return new_stmt, stmt
      return '', ''
    elif expression_type == 'NamespaceDecl':
      found, new_stmt = mark_string_pos(stmt, AMAP_NS_STRING_LIST, '|+|')
      if found:
        new_stmt = new_stmt.replace('|+|', AMAP_NS_NEW_STRING)
        return new_stmt, stmt
      return '', ''
    elif expression_type == 'NamedDecl':
      # if self.skip_stmt(stmt):
      #   return '', ''
      if stmt.startswith('(lambda ') or stmt.startswith('~(lambda ') or stmt.startswith('operator '):
        return '', ''
      found, new_stmt = mark_string_pos(stmt, AMAP_CLS_STRING_LIST, '|-|')
      if found:
        new_stmt = new_stmt.replace('|-|', AMAP_CLS_NEW_STRING)
        return new_stmt, stmt
      return '', ''
    elif expression_type == 'CXXRecordDecl':
      found, new_stmt = mark_string_pos(stmt, AMAP_CLS_STRING_LIST, '|-|')
      if found:
        new_stmt = new_stmt.replace('|-|', AMAP_CLS_NEW_STRING)
        return new_stmt, stmt
      return '', ''
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
      found, new_stmt = mark_string_pos(stmt_list[0], AMAP_CLS_STRING_LIST, '|-|')
      if found:
        new_stmt = new_stmt.replace('|-|', AMAP_CLS_NEW_STRING)
        return new_stmt, stmt_list[0]
      return '', ''
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
      found_ns, new_stmt = mark_all_string_pos(new_stmt, AMAP_NS_STRING_LIST, '|+|', '', '::')
      found_cls, new_stmt = mark_all_string_pos(new_stmt, AMAP_CLS_STRING_LIST, '|-|')
      if found_ns or found_cls:
        new_stmt = new_stmt.replace('|+|', f'{AMAP_NS_NEW_STRING}::')
        new_stmt = new_stmt.replace('|-|', AMAP_CLS_NEW_STRING)
        return new_stmt, stmt
      return '', ''
    elif expression_type == 'UsingDirectiveDecl':
      namespace_pattern = r'using\s+namespace\s+'
      namespace_pos = re.search(namespace_pattern, stmt)
      found_ns = False
      found_cls = False
      if namespace_pos:
        new_stmt = stmt
        found_ns, new_stmt = mark_all_string_pos(new_stmt, AMAP_NS_STRING_LIST, '|+|')
        if found_ns:
          new_stmt = new_stmt.replace('|+|', f'{AMAP_NS_NEW_STRING}')
          return new_stmt, stmt
      else:
        found_ns, new_stmt = mark_all_string_pos(new_stmt, AMAP_NS_STRING_LIST, '|+|', '', '::')
        found_cls, new_stmt = mark_all_string_pos(new_stmt, AMAP_CLS_STRING_LIST, '|-|')
        if found_ns or found_cls:
          new_stmt = new_stmt.replace('|+|', f'{AMAP_NS_NEW_STRING}::')
          new_stmt = new_stmt.replace('|-|', AMAP_CLS_NEW_STRING)
          return new_stmt, stmt
      return '', ''
    else:
      LOG('error', f'unchecked expression type: {expression_type}')
    return None, None

  def check_log_entry(self, log_entry, log_entry_type, append_type = ''):
    if log_entry['type'] != log_entry_type:
      return
    file_path = log_entry['file']
    if len(file_path) == 0:
      return
    if skip_replace_file(file_path):
      return
    file_line = log_entry['line']
    file_column = log_entry['column']
    stmt = log_entry['stmt']
    expression_type = log_entry['exptype']
    dfile_path = log_entry['dfile']
    new_stmt, adj_stmt = self.match_expression(stmt, expression_type + append_type, dfile_path)
    if new_stmt is not None:
      if len(new_stmt) == 0:
        return
      file_column += stmt.find(adj_stmt)
      stmt = adj_stmt
      if not self.match_update_position(file_path, file_line, file_column, stmt, expression_type + append_type):
        self.cantidate_replace_files[file_path].update_position[(file_line, file_column)] = SourceCodeUpdatePosition(file_line, file_column, log_entry_type + ":" + expression_type + append_type, stmt, new_stmt)
      return

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
          sorted_positions = sorted(update_info.update_position.values(), key=lambda pos: (pos.line, pos.column))
          replace_line_number = -1
          replace_diff = 0
          line_content = ''
          for update_position in sorted_positions:
            if update_position.type == 'expression':
              if replace_line_number != update_position.line - 1:
                replace_line_number = update_position.line - 1
                line_content = lines[replace_line_number]
                replace_diff = 0
              stmt = update_position.stmt
              new_stmt = update_position.tostmt
              multi_lines = new_stmt.rstrip('\n').count('\n')
              for index in range(1, multi_lines + 1):
                line_content += lines[replace_line_number + index]
                index = index + 1
              column = update_position.column + replace_diff - 1
              src_stmt = line_content[column:column + len(stmt)]
              if src_stmt != stmt:
                LOG('error', f'{update_position.kind} src_stmt != stmt [[{src_stmt}]] != [[{stmt}]]')
                continue
              line_content = line_content[0:column] + new_stmt + line_content[column + len(stmt):]
              new_line_parts = line_content.splitlines(keepends=True)
              lines[replace_line_number:replace_line_number + len(new_line_parts)] = new_line_parts
              replace_diff = replace_diff + len(new_stmt) - len(stmt)
              LOG('info', f'replace {update_position.kind} {update_position.line}:{update_position.column} [[{stmt}]] -> [[{new_stmt}]] [[{line_content}]]')
            # elif update_position.type == 'header':
            #   continue
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
      ('FunctionDecl', ''),
      ('NamedDecl', ''),
      ('CXXRecordDecl', ''),
      ('CallExpr', ''),
      ('DeclRefExpr', '='),
      ('TypeLoc', '+'),
      ('NamespaceDecl', ''),
      ('UsingDirectiveDecl', '')
    ]
    if True:
      for analyze_type, append_type in analyze_types:
        thread = threading.Thread(target=self.analyze_source_files, args=(analyze_type,), kwargs={'append_type': append_type, 'reuse_analyzed_result': reuse_analyzed_result})
        threads.append(thread)
        thread.start()
      thread = threading.Thread(target=self.analyze_source_files_MacroDefExp, args=(reuse_analyzed_result,))
      threads.append(thread)
      thread.start()
      for thread in threads:
        thread.join()
    else:
      self.analyze_source_files('FunctionDecl', reuse_analyzed_result=reuse_analyzed_result)
      self.analyze_source_files('NamedDecl', reuse_analyzed_result=reuse_analyzed_result)
      self.analyze_source_files('CXXRecordDecl', reuse_analyzed_result=reuse_analyzed_result)
      self.analyze_source_files('CallExpr', reuse_analyzed_result=reuse_analyzed_result)
      self.analyze_source_files('DeclRefExpr', reuse_analyzed_result=reuse_analyzed_result, append_type='=')
      self.analyze_source_files('TypeLoc', reuse_analyzed_result=reuse_analyzed_result, append_type='+')
      self.analyze_source_files('NamespaceDecl', reuse_analyzed_result=reuse_analyzed_result)
      self.analyze_source_files('UsingDirectiveDecl', reuse_analyzed_result=reuse_analyzed_result)
      self.analyze_source_files_MacroDefExp(reuse_analyzed_result=reuse_analyzed_result)
    self.replace_in_source_files()

if __name__ == "__main__":
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
