import os
import argparse
import re
import json
# import clang.cindex
import logging
import sys

'''
此脚本在给定的根目录中搜索所有名为 'Export_Inc' 的目录
然后将这些目录中所有包含 'AMap' 的文件重命名为用户指定的新名称
此外，它还会更新所有C、C++源码文件中的 include 语句，以反映新的文件名
'''

# AMAP_STRING_LIST = ['AMap', 'Amap', 'amap', 'AMAP']
AMAP_STRING_LIST = ['LbsAmap', 'NS']

# clang.cindex.Config.set_library_file('/usr/lib/x86_64-linux-gnu/libclang-17.so.17')
log_file = 'update_sourcecode.log'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_file, filemode='w')
analyzer = '/share/dev/test_cpp/build/analyzer'
reuse_analyzed_result = True

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

class SourceCodeUpdater:
  def __init__(self, compile_commands_json, input_directory, output_directory, new_name):
    self.compile_commands_json = compile_commands_json
    self.source_files = {}
    self.parse_compile_commands()
    self.input_directory = input_directory
    self.output_directory = output_directory
    self.new_name = new_name
    if not os.path.exists(self.input_directory):
      print(f"错误: 输入目录 '{self.input_directory}' 不存在", file=sys.stderr)
      exit(1)
    if os.path.exists(self.output_directory):
      logging.warning(f"输出目录 '{self.output_directory}' 已存在")
    # self.index = clang.cindex.Index.create()
    self.cantidate_replace_files = {}
    self.rename_file_list = []
    self.source_files_dependencies = {}

  def parse_compile_commands(self):
    self.include_dirs = []
    with open(self.compile_commands_json, 'r') as f:
      compile_commands = json.load(f)
    for entry in compile_commands:
      command = entry['command']
      compile_commands = command.split(' -o ')[0]
      source_file = os.path.abspath(entry['file'])
      if 'ThirdPartyLib' in source_file:
        continue
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
      logging.info(f"分析{cmd}: {file_path}")
      compile_commands = self.source_files.get(file_path)
      if compile_commands is None:
        result = os.system(f'echo "# {file_path}" >> {analyzer_log_file} && ANALYZE_CMD="{cmd}" {analyzer} {file_path} -p {os.path.dirname(self.compile_commands_json)} >> {analyzer_log_file}')
      else:
        result = os.system(f'echo "# {file_path}" >> {analyzer_log_file} && ANALYZE_CMD="{cmd}" {analyzer} {file_path} -- {compile_commands} -Wno-everything >> {analyzer_log_file}')
      if result != 0:
        print(f"错误: 分析源文件时出错，返回值: {result} {file_path}", file=sys.stderr)
        exit(1)

  def analyze_source_files_InclusionDirective(self):
    analyzer_log_file = 'analyzer_InclusionDirective.log'
    self.exec_analyzer(analyzer_log_file, "InclusionDirective|", self.source_files.keys(), -1, reuse_analyzed_result)
    with open(analyzer_log_file, 'r') as analyzer_log_file_content:
      for line in analyzer_log_file_content:
        if line.startswith('#'):
          continue
        try:
          log_entry = json.loads(line)
          if log_entry['type'] != 'InclusionDirective':
            continue
          filepath = log_entry['file']
          includefilepath = log_entry['includefilepath']
          if filepath == includefilepath:
            continue
          if self.source_files_dependencies.get(includefilepath) is None:
            self.source_files_dependencies[includefilepath] = []
          if filepath not in self.source_files_dependencies[includefilepath]:
            self.source_files_dependencies[includefilepath].append(filepath)
        except json.JSONDecodeError as e:
          print(f"错误: 解析include分析日志时出错: {e} {line}", file=sys.stderr)
          exit(1)

  def match_macro(self, macroname):
    if macroname == 'BEGIN_NS':
      return f'namespace {self.new_name} {{'
    if macroname == 'END_NS':
      return f'}}'
    return None

  def analyze_source_files_MacroExpands(self):
    analyzer_log_file = 'analyzer_MacroExpands.log'
    self.exec_analyzer(analyzer_log_file, "MacroExpands|", self.source_files.keys(), -1, reuse_analyzed_result)
    with open(analyzer_log_file, 'r') as analyzer_log_file_content:
      for line in analyzer_log_file_content:
        if line.startswith('#'):
          continue
        try:
          log_entry = json.loads(line)
          if log_entry['type'] != 'MacroExpands':
            continue
          file_path = log_entry['file']
          file_line = log_entry['line']
          file_column = log_entry['column']
          macroname = log_entry['macroname']
          matchupdate = self.match_macro(macroname)
          if matchupdate :
            if file_path not in self.cantidate_replace_files:
              self.cantidate_replace_files[file_path] = SourceCodeUpdateInfo(file_path)
            position_key = (file_line, file_column)
            if position_key not in self.cantidate_replace_files[file_path].update_position:
              self.cantidate_replace_files[file_path].update_position[position_key] = SourceCodeUpdatePosition(file_line, file_column, 'MacroExpands', macroname, matchupdate)
        except json.JSONDecodeError as e:
          print(f"错误: 解析include分析日志时出错: {e} {line}", file=sys.stderr)
          exit(1)

  def analyze_source_files_NamespaceDecl(self):
    analyzer_log_file = 'analyzer_NamespaceDecl.log'
    self.exec_analyzer(analyzer_log_file, "NamespaceDecl|", self.source_files.keys(), -1, reuse_analyzed_result)
    with open(analyzer_log_file, 'r') as analyzer_log_file_content:
      for line in analyzer_log_file_content:
        if line.startswith('#'):
          continue
        try:
          log_entry = json.loads(line)
          if log_entry['type'] != 'NamespaceDecl':
            continue
          file_path = log_entry['file']
          file_line = log_entry['line']
          file_column = log_entry['column']
          namespace = log_entry['stmt']
          if any(amap_string in namespace for amap_string in AMAP_STRING_LIST):
            if file_path not in self.cantidate_replace_files:
              self.cantidate_replace_files[file_path] = SourceCodeUpdateInfo(file_path)
            position_key = (file_line, file_column)
            if position_key not in self.cantidate_replace_files[file_path].update_position:
              self.cantidate_replace_files[file_path].update_position[position_key] = SourceCodeUpdatePosition(file_line, file_column, 'NamespaceDecl', namespace)
            # return
        except json.JSONDecodeError as e:
          print(f"错误: 解析NamespaceDecl分析日志时出错: {e} {line}", file=sys.stderr)
          exit(1)
    return

  def match_expression(self, amap_string, stmt, expression_type):
    if expression_type == 'Elaborated':
      pattern = re.compile(r'^' + re.escape(amap_string) + r'(::)?')
      pos = pattern.search(stmt)
      if pos is not None:
        return stmt[pos.start():pos.end()]
    elif expression_type == 'Enum':
      pattern = re.compile(r'^' + re.escape(amap_string) + r'::')
      pos = pattern.search(stmt)
      if pos is not None:
        return stmt[pos.start():pos.end()]
    elif expression_type == 'FunctionProto':
      pattern = re.compile(r'^' + re.escape(amap_string) + r'::')
      pos = pattern.search(stmt)
      if pos is not None:
        return stmt[pos.start():pos.end()]
    elif expression_type == 'FunctionImpl':
      pattern = re.compile(r'^' + re.escape(amap_string) + r'::')
      pos = pattern.search(stmt)
      if pos is not None:
        return stmt[pos.start():pos.end()]
    elif expression_type == 'UsingDirectiveDecl':
      if amap_string in stmt:
        return stmt
    return None
  
  def check_log_entry(self, log_entry, log_entry_type):
    if log_entry['type'] != log_entry_type:
      return
    file_path = log_entry['file']
    file_line = log_entry['line']
    file_column = log_entry['column']
    stmt = log_entry['stmt']
    expression_type = log_entry['exptype']
    dfile_path = log_entry['dfile']
    dfile_line = log_entry['dline']
    dfile_column = log_entry['dcolumn']
    dstmt = log_entry['dstmt']
    found = False
    for amap_string in AMAP_STRING_LIST:
      new_stmt = self.match_expression(amap_string, stmt, expression_type)
      if new_stmt:
        found = True
        if file_path not in self.cantidate_replace_files:
          self.cantidate_replace_files[file_path] = SourceCodeUpdateInfo(file_path)
        position_key = (file_line, file_column)
        if position_key not in self.cantidate_replace_files[file_path].update_position:
          self.cantidate_replace_files[file_path].update_position[position_key] = SourceCodeUpdatePosition(file_line, file_column, log_entry_type, new_stmt)
    if found:
      return
    for amap_string in AMAP_STRING_LIST:
      new_stmt = self.match_expression(amap_string, dstmt, expression_type)
      if new_stmt:
        if dfile_path not in self.cantidate_replace_files:
          self.cantidate_replace_files[dfile_path] = SourceCodeUpdateInfo(dfile_path)
        position_key = (dfile_line, dfile_column)
        if position_key not in self.cantidate_replace_files[dfile_path].update_position:
          self.cantidate_replace_files[dfile_path].update_position[position_key] = SourceCodeUpdatePosition(dfile_line, dfile_column, log_entry_type, new_stmt)

  def analyze_source_files_TypeLoc(self):
    analyzer_log_file = 'analyzer_TypeLoc.log'
    self.exec_analyzer(analyzer_log_file, "TypeLoc|", self.source_files.keys(), -1, reuse_analyzed_result)
    with open(analyzer_log_file, 'r') as analyzer_log_file_content:
      for line in analyzer_log_file_content:
        if line.startswith('#'):
          continue
        try:
          log_entry = json.loads(line)
          self.check_log_entry(log_entry, 'TypeLoc')
          # return
        except json.JSONDecodeError as e:
          print(f"错误: 解析TypeLoc分析日志时出错: {e} {line}", file=sys.stderr)
          exit(1)
    return

  def analyze_source_files_DeclRefExpr(self):
    analyzer_log_file = 'analyzer_DeclRefExpr.log'
    self.exec_analyzer(analyzer_log_file, "DeclRefExpr|", self.source_files.keys(), -1, reuse_analyzed_result)
    with open(analyzer_log_file, 'r') as analyzer_log_file_content:
      for line in analyzer_log_file_content:
        if line.startswith('#'):
          continue
        try:
          log_entry = json.loads(line)
          self.check_log_entry(log_entry, 'DeclRefExpr')
          # return
        except json.JSONDecodeError as e:
          print(f"错误: 解析DeclRefExpr分析日志时出错: {e} {line}", file=sys.stderr)
          exit(1)
    return

  def analyze_source_files_FunctionDecl(self):
    analyzer_log_file = 'analyzer_FunctionDecl.log'
    self.exec_analyzer(analyzer_log_file, "FunctionDecl|", self.source_files.keys(), -1, reuse_analyzed_result)
    with open(analyzer_log_file, 'r') as analyzer_log_file_content:
      for line in analyzer_log_file_content:
        if line.startswith('#'):
          continue
        try:
          log_entry = json.loads(line)
          self.check_log_entry(log_entry, 'FunctionDecl')
          # return
        except json.JSONDecodeError as e:
          print(f"错误: 解析FunctionDecl分析日志时出错: {e} {line}", file=sys.stderr)
          exit(1)
    return

  def analyze_source_files_UsingDirectiveDecl(self):
    # 分析namespace
    analyzer_log_file = 'analyzer_UsingDirectiveDecl.log'
    self.exec_analyzer(analyzer_log_file, "UsingDirectiveDecl|", self.source_files.keys(), -1, reuse_analyzed_result)
    with open(analyzer_log_file, 'r') as analyzer_log_file_content:
      for line in analyzer_log_file_content:
        if line.startswith('#'):
          continue
        try:
          log_entry = json.loads(line)
          self.check_log_entry(log_entry, 'UsingDirectiveDecl')
          # return
        except json.JSONDecodeError as e:
          print(f"错误: 解析UsingDirectiveDecl分析日志时出错: {e} {line}", file=sys.stderr)
          exit(1)
    return

  def replace_statement(self, statement):
    replaced = statement
    for amap_string in AMAP_STRING_LIST:
      if amap_string in replaced:
        replaced = replaced.replace(amap_string, self.new_name)
    return replaced

  def replace_namespace_in_source_files(self):
    for source_file, update_info in self.cantidate_replace_files.items():
      output_source_file = os.path.join(self.output_directory, os.path.relpath(source_file, self.input_directory))
      logging.info(f"from file: {source_file}")
      logging.info(f"to file: {output_source_file}")
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
              new_stmt = ''
              if update_position.tostmt:
                new_stmt = update_position.tostmt
              else:
                new_stmt = self.replace_statement(stmt)
              column = update_position.column + replace_diff - 1
              line_content = line_content[0:column] + new_stmt + line_content[column + len(stmt):]
              lines[replace_line_number] = line_content
              replace_diff = replace_diff + len(new_stmt) - len(stmt)
              logging.info(f'replace {update_position.kind} {update_position.line}:{update_position.column} [[{stmt}]] -> [[{new_stmt}]] [[{line_content}]]')
            elif update_position.type == 'header':
              continue
          os.makedirs(os.path.dirname(output_source_file), exist_ok=True)
          with open(output_source_file, 'w') as output_source:
            output_source.writelines(lines)

  # def rename_header_files_in_directory(self, directory):
  #   for dirpath, dirnames, filenames in os.walk(directory):
  #     if self.ignore(dirpath):
  #       continue
  #     for filename in filenames:
  #       if AMAP_STRING in filename and (filename.endswith('.h') or filename.endswith('.hpp')):
  #         new_filename = filename.replace(AMAP_STRING, self.new_name)
  #         logging.info(f'cd {dirpath} && mv {filename} {new_filename} && cd -')
  #         result = os.system(f'cd {dirpath} && mv {filename} {new_filename} && cd -')
  #         if result != 0:
  #           print(f"错误: 移动文件时出错，返回值: {result} {filename} => {new_filename}", file=sys.stderr)
  #           exit(1)
  #         logging.info(f"rename header file: {filename} {new_filename}")
  #         self.rename_file_list.append((filename, new_filename))

  # def rename_sourcefile(self):
  #   for dirpath, dirnames, filenames in os.walk(self.output_directory):
  #     logging.info(f'Processing directory: {dirpath}')
  #     if self.need_replace(dirpath):
  #       self.rename_header_files_in_directory(dirpath)

  # def replace_include_in_files(self, old_include, new_include):
  #   for item in self.source_files:
  #     file_path = os.path.join(self.output_directory, os.path.relpath(item[0], self.input_directory))
  #     replaced = False
  #     content = ''
  #     with open(file_path, 'r') as file:
  #       content = file.read()
  #       includes = re.findall(r'#\s*include\s*["<](.*?)[">]', content, re.DOTALL)
  #       for include in includes:
  #         if old_include in include:
  #           new_include_path = include.replace(old_include, new_include)
  #           content = content.replace(include, new_include_path)
  #           replaced = True
  #       if not replaced:
  #         continue
  #     with open(file_path, 'w') as file:
  #       file.write(content)

  # def replace_sourcefile_include(self):
  #   for old_name, new_name in self.rename_file_list:
  #     self.replace_include_in_files(old_name, new_name)

  def copy_source_files(self):
    if os.path.exists(self.output_directory):
      print(f"错误: 输出目录 '{self.output_directory}' 已存在", file=sys.stderr)
      exit(1)
    logging.info(f'cp -r {self.input_directory} {self.output_directory}')
    result = os.system(f'cp -r {self.input_directory} {self.output_directory}')
    if result != 0:
      print(f"错误: 复制目录时出错，返回值: {result} {self.input_directory} {self.output_directory}", file=sys.stderr)
      exit(1)

  def process_source_files(self):
    # self.copy_source_files()
    self.analyze_source_files_MacroExpands()
    self.analyze_source_files_NamespaceDecl()
    self.analyze_source_files_TypeLoc()
    self.analyze_source_files_DeclRefExpr()
    self.analyze_source_files_FunctionDecl()
    self.analyze_source_files_UsingDirectiveDecl()
    self.replace_namespace_in_source_files()
    # self.analyze_source_files_InclusionDirective()
    # self.rename_sourcefile()
    # self.replace_sourcefile_include()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('compile_commands_json', type=str, help='Path to compile_commands.json')
  parser.add_argument('input_directory', type=str, help='The root directory to search for Export_Inc directories')
  parser.add_argument('output_directory', type=str, help='The directory to save the updated files')
  parser.add_argument('new_name', type=str, help='The new name to replace AMap with')
  parser.add_argument('extra_cmd', type=str, help='')
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
  if not args.new_name:
    print("错误: 必须指定 new_name 参数", file=sys.stderr)
    exit(1)
  if args.extra_cmd:
    logging.info(f'{args.extra_cmd}')
    result = os.system(f'{args.extra_cmd}')
    if result != 0:
      print(f"错误: 执行命令时出错，返回值: {result} {args.extra_cmd}", file=sys.stderr)
      exit(1)

  updater = SourceCodeUpdater(args.compile_commands_json, args.input_directory, args.output_directory, args.new_name)
  updater.process_source_files()
  print("完成")
