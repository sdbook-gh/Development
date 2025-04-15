#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclAccessPair.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Lex/MacroArgs.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <vector>

using namespace clang;
using namespace clang::tooling;
using namespace clang::ast_matchers;

/**
 * @brief 获取输入路径的真实绝对路径
 *
 * 此函数调用系统函数 realpath() 将给定的输入路径转换为标准的绝对路径。
 * 如果转换成功，返回转换后的绝对路径；如果失败（例如路径不存在或权限不足），则返回空字符串。
 *
 * @param path 输入的文件路径，可以是相对路径或包含符号链接的路径
 * @return std::string 转换后的绝对路径；若转换失败返回空字符串
 */
static std::string getPath(const std::string &path) {
  char *realPathPtr = realpath(path.c_str(), nullptr);
  if (realPathPtr == nullptr)
    return "";
  std::string realPath{realPathPtr};
  free(realPathPtr);
  return realPath;
}

/**
 * @brief 跳过的路径集合。
 *
 * 该静态向量用于存储所有需要在分析过程中跳过的文件路径，
 * 避免对这些路径下的文件进行额外的处理或分析。
 */
static std::vector<std::string> skipPaths;

/**
 * @brief 判断给定路径是否需要被跳过
 *
 * 该函数用于检查传入的实际路径（realPath）是否符合跳过处理的条件。
 *
 * @param realPath 待检查的实际路径字符串
 * @return bool 若路径符合跳过条件则返回 true，否则返回 false
 */
static bool skipPath(const std::string &realPath) {
  if (realPath.empty())
    return true;
  if (realPath.find("<") == 0) {
    return true;
  }
  for (const auto &skipPath : skipPaths) {
    if (realPath.find(skipPath) != std::string::npos)
      return true;
  }
  return false;
}

/**
 * @brief 转义输入字符串中的特殊字符以符合 JSON 格式要求。
 *
 * 该函数遍历输入字符串中的每个字符，并对可能影响 JSON 格式的字符进行转义：
 * - 对双引号 (") 进行转义为 \"。
 * - 对反斜杠 (\) 进行转义为 \\。
 * - 对退格符 (\b)、换页符 (\f)、换行符 (\n)、回车符 (\r) 和制表符 (\t)
 * 进行转义。
 * - 对 ASCII 码值在 0x00 到 0x1F 范围内的其他控制字符，转换为 Unicode
 * 转义序列，例如 \uXXXX。
 *
 * @param input 要转义的原始字符串
 * @return 转义后的字符串，可安全用于 JSON 字符串构造中。
 */
static std::string escapeJsonString(const std::string &input) {
  const std::string &input_string = input;
  std::ostringstream ss;
  for (char c : input_string) {
    switch (c) {
    case '"':
      ss << "\\\"";
      break;
    case '\\':
      ss << "\\\\";
      break;
    case '\b':
      ss << "\\b";
      break;
    case '\f':
      ss << "\\f";
      break;
    case '\n':
      ss << "\\n";
      break;
    case '\r':
      ss << "\\r";
      break;
    case '\t':
      ss << "\\t";
      break;
    default:
      if ('\x00' <= c && c <= '\x1f') {
        ss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)c;
      } else {
        ss << c;
      }
    }
  }
  return ss.str();
}

/**
 * @brief 检查文件中指定位置的内容是否与给定文本匹配.
 *
 * 函数读取指定文件的所有行，并从指定的起始行和起始列开始逐行比较字符，
 * 检查文件中连续的文本是否与输入的文本一致。每一行的比较依据给定文本的相应行长度进行。
 * 如果任一条件不满足（如文件路径为空、起始位置无效、文件无法打开、匹配的文本为空等），
 * 函数将返回 false。
 *
 * @param filePath 文件路径，不能为空。
 * @param startLine 匹配开始的行号（1-based），必须在文件行数范围内且不小于1。
 * @param startColumn
 * 匹配开始的列号（1-based），必须在对应行字符范围内且不小于1。
 * @param text 需要在文件中匹配的文本，不能为空。
 * @return 如果指定位置的内容与给定文本完全匹配，则返回 true，否则返回 false。
 */
static bool matchFileContent(const std::string &filePath, int startLine,
                             int startColumn, const std::string &text) {
  if (filePath.empty() || startLine < 1 || startColumn < 1)
    return false;
  if (text.empty())
    return false;
  std::fstream file(filePath);
  if (!file)
    return false;
  std::vector<std::string> fileLines;
  std::string line;
  while (std::getline(file, line))
    fileLines.emplace_back(line);
  std::stringstream iss(text);
  std::vector<std::string> textLines;
  std::string textLine;
  while (std::getline(iss, textLine, '\n'))
    textLines.emplace_back(textLine);
  if (startLine < 1 || startLine > static_cast<int>(fileLines.size()))
    return false;
  if (startColumn < 1 ||
      startColumn > static_cast<int>(fileLines[startLine - 1].size()) + 1)
    return false;
  for (size_t i = 0; i < textLines.size(); ++i) {
    int fileIndex = startLine - 1 + i;
    const std::string &fileLine = fileLines[fileIndex];
    std::string sub;
    if (i == 0) {
      sub = fileLine.substr(startColumn - 1, textLines[i].size());
    } else {
      sub = fileLine.substr(0, textLines[i].size());
    }
    if (sub != textLines[i])
      return false;
  }
  return true;
}

/**
 * @brief 从指定文件中提取当前指定位置之前的一段代码片段。
 *
 * 根据给定的文件路径、起始行号、起始列号及待匹配文本（作为正则表达式），
 * 本函数逐行向上搜索文件内容，查找最后一次出现该文本的地方，并截取从该位置到当前起始位置之间的代码片段。
 *
 * 具体流程：
 * 1. 检查文件路径和起始位置是否合法，若无效则返回空字符串；
 * 2. 根据文件路径打开文件并读取所有行内容；
 * 3. 对于起始行（以及之前的行），使用正则表达式寻找文本的最后一次匹配位置；
 * 4.
 * 检查匹配位置后面是否紧跟分号（';'），若存在且分号位置在匹配文本之后，则中断搜索并返回空字符串；
 * 5. 如果在多个行中找到了匹配文本，则截取从匹配位置到起始位置之间的代码片段；
 * 6. 如果提供了 searchLine 和 searchColumn
 * 参数，则将匹配到的行号和列号（均以1为基准）返回。
 *
 * @param filePath       待读取文件的完整路径。
 * @param startLine      起始行号（从1开始）。
 * @param startColumn    起始列号（从1开始）。
 * @param text           用于匹配的正则表达式模式字符串。
 * @param searchLine
 * 可选参数，若不为nullptr，则存放匹配文本所在行号（从1开始）。
 * @param searchColumn
 * 可选参数，若不为nullptr，则存放匹配文本的起始列号（从1开始）。
 *
 * @return std::string
 * 成功则返回提取的代码片段；如果未找到匹配项或遇到错误，返回空字符串。
 */
std::string getStmtBefore(const std::string &filePath, int startLine,
                          int startColumn, const std::string &text,
                          int *searchLine = nullptr,
                          int *searchColumn = nullptr) {
  auto regex_rfind = [](const std::string &input,
                        const std::string &pattern) -> std::size_t {
    std::regex re(pattern);
    std::sregex_iterator iter(input.begin(), input.end(), re);
    std::sregex_iterator end;
    std::size_t lastPos = std::string::npos;
    for (; iter != end; ++iter) {
      lastPos = iter->position();
    }
    return lastPos;
  };
  if (filePath.empty() || startLine < 1 || startColumn < 1)
    return "";
  if (text.empty())
    return "";
  std::fstream file(filePath);
  if (!file)
    return "";
  std::vector<std::string> fileLines;
  std::string line;
  while (std::getline(file, line))
    fileLines.emplace_back(line);
  int startLineIndex = startLine - 1;
  int startColIndex = startColumn - 1;
  int foundLine = -1;
  size_t foundCol = std::string::npos;
  size_t foundStop = std::string::npos;
  for (int i = startLineIndex; i >= 0; --i) {
    const std::string &currentLine = fileLines[i];
    if (i == startLineIndex) {
      std::string prefix = currentLine.substr(0, startColIndex);
      foundCol = regex_rfind(prefix, text);
      foundStop = prefix.rfind(";");
      if (foundCol != std::string::npos) {
        if (foundStop != std::string::npos && foundStop >= foundCol) {
          return "";
        }
        foundLine = i;
        break;
      }
    } else {
      foundCol = regex_rfind(currentLine, text);
      foundStop = currentLine.rfind(";");
      if (foundCol != std::string::npos) {
        if (foundStop != std::string::npos && foundStop >= foundCol) {
          return "";
        }
        foundLine = i;
        break;
      }
    }
  }
  if (foundLine < 0)
    return "";
  std::string result;
  if (foundLine == startLineIndex) {
    result = fileLines[foundLine].substr(foundCol, startColIndex - foundCol);
  } else {
    result = fileLines[foundLine].substr(foundCol);
    for (int j = foundLine + 1; j < startLineIndex; ++j) {
      result += "\n" + fileLines[j];
    }
    result += "\n" + fileLines[startLineIndex].substr(0, startColIndex);
  }
  if (searchLine != nullptr && searchColumn != nullptr) {
    *searchLine = foundLine + 1;
    *searchColumn = foundCol + 1;
  }
  return result;
}

/**
 * @brief 宏匹配信息结构体
 *
 * 该结构体用于存储在预处理过程中匹配到的宏信息，包括其定义位置以及相关语句内容。
 *
 * 成员说明：
 * - line：宏定义所在的行号（从1开始计数）。
 * - column：宏定义所在的列号（从1开始计数）。
 * - name：宏的名称。
 * - stmt：宏定义的完整语句文本，通常为经过转义以适应JSON格式的字符串。
 */
struct MacroMatchInfo {
  int line{0};
  int column{0};
  std::string name;
  std::string stmt;
};
std::map<std::string, MacroMatchInfo> macroDefinition_map;
/**
 * @brief 判断指定位置是否位于宏定义内部
 *
 * 此函数检查给定的文件路径以及指定的行号和列号对应的位置，
 * 是否处于预先记录的宏定义区域内。如果该位置在某个宏定义的范围内，
 * 则返回该宏定义的名称；否则返回空字符串，表示该位置不在任何宏定义内部。
 *
 * @param file_path 文件的实际路径字符串
 * @param line 行号（从1开始计数）
 * @param column 列号（从1开始计数）
 * @return std::string
 * 如果指定位置处于宏定义内部，则返回宏名称，否则返回空字符串
 */
std::string isInsideMacroDefinition(const std::string &file_path, int line,
                                    int column) {
  if (macroDefinition_map.count(file_path) > 0) {
    const MacroMatchInfo &definition = macroDefinition_map[file_path];
    std::stringstream iss(definition.stmt);
    std::string textLine;
    int match_line = definition.line, match_column = definition.column,
        index = 0;
    while (std::getline(iss, textLine, '\n')) {
      if (line == match_line && column >= match_column &&
          column < match_column + textLine.size()) {
        return definition.name;
      } else if (index == 0) {
        match_column = 0;
      }
      match_line++;
      index++;
    }
  }
  return "";
}

static std::string option;

class AnalysisPPCallback : public PPCallbacks {
private:
  Preprocessor &PP;
  struct MDInfo {
    std::string file;
    std::string stmt;
  };
  std::vector<MDInfo> MD_vec;

public:
  AnalysisPPCallback(Preprocessor &PP) : PP(PP) {}

  /**
   * @brief 处理宏定义事件的回调函数。
   *
   * 该函数根据全局选项判断当前处理逻辑：
   * - 如果选项包含 "SkipMacroDef|"：
   *   - 检查宏名相关的宏令牌(MacroNameTok)的位置是否有效且不在系统头文件中。
   *   - 利用 SourceManager 获取文件路径、行号及列号信息，并对宏名称进行 JSON
   * 格式字符串转义。
   *   -
   * 如果宏名称为空或已在跳过列表中，则直接返回；否则，如果文件路径被设定为跳过，则打印一条类型为
   * "SkipMacroDef" 的 JSON 信息。
   *
   * - 如果选项包含 "MacroDefExpIfndefInclusionCommentSkip|"：
   *   - 同样先验证位置有效性和是否在系统头文件内。
   *   - 取得宏定义的起始位置和结束位置，利用 Lexer
   * 获取宏定义的原始文本，并对文本进行适当格式化处理后转义。
   *   - 根据文件路径和内容匹配情况，可能在不同情况下打印不同前缀（例如 "##" 或
   * "####"）。
   *   -
   * 在满足匹配条件后，还会将构造的宏定义信息(包括行号、列号、宏名称和原始宏文本)存储到一个映射中，
   *     或者将组合后的语句存入一个用于后续检查的向量中。
   *   - 最后，根据处理结果输出类型为 "MacroDefined" 的 JSON
   * 格式信息，其中包含文件名、位置、宏名称以及对应的宏文本。
   *
   * @param MacroNameTok 用于标识宏名称的令牌对象。
   * @param MD 指向包含宏相关信息的 MacroDirective 对象的指针。
   */
  void MacroDefined(const Token &MacroNameTok,
                    const MacroDirective *MD) override {
    if (option.find("SkipMacroDef|") != std::string::npos) {
      const MacroInfo *MI = MD->getMacroInfo();
      SourceManager &SM = PP.getSourceManager();
      SourceLocation Loc = MacroNameTok.getLocation();
      if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
        return;
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      std::string file = getPath(PLoc.getFilename());
      int line = PLoc.getLine();
      int column = PLoc.getColumn();
      std::string stmt =
          escapeJsonString(MacroNameTok.getIdentifierInfo()->getName().str());
      static std::set<std::string> skipMacroDefSet;
      std::stringstream ss;
      if (stmt.size() == 0 || skipMacroDefSet.count(stmt) > 0) {
        return;
      } else if (skipPath(file)) {
        ss << "{\"type\":\"SkipMacroDef\",\"file\":\"" << file
           << "\",\"line\":" << line << ",\"column\":" << column
           << ",\"stmt\":\"" << stmt << "\"}";
        printf("%s\n", ss.str().c_str());
      }
    } else if (option.find("MacroDefExpIfndefInclusionCommentSkip|") !=
               std::string::npos) {
      const MacroInfo *MI = MD->getMacroInfo();
      SourceManager &SM = PP.getSourceManager();
      SourceLocation Loc = MacroNameTok.getLocation();
      if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
        return;
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      std::string file = getPath(PLoc.getFilename());
      int line = PLoc.getLine();
      int column = PLoc.getColumn();
      CharSourceRange CSR = CharSourceRange::getCharRange(
          MI->getDefinitionLoc(),
          Lexer::getLocForEndOfToken(MI->getDefinitionEndLoc(), 0, SM,
                                     PP.getLangOpts()));
      std::string raw_stmt =
          Lexer::getSourceText(CSR, SM, PP.getLangOpts()).str();
      std::string stmt = escapeJsonString(raw_stmt);
      std::string name =
          escapeJsonString(MacroNameTok.getIdentifierInfo()->getName().str());
      SourceRange expansionRange(MI->getDefinitionLoc(),
                                 MI->getDefinitionEndLoc());
      std::stringstream ss;
      if (skipPath(file)) {
        ss << "##";
      } else if (!matchFileContent(file, line, column, raw_stmt)) {
        ss << "####";
      } else {
        macroDefinition_map[file] =
            MacroMatchInfo{line, column, name, raw_stmt};
        std::string new_stmt = getStmtBefore(file, line, column, "#");
        if (!new_stmt.empty()) {
          MD_vec.push_back({file, new_stmt + raw_stmt});
        }
      }
      ss << "{\"type\":\"MacroDefined\",\"file\":\"" << file
         << "\",\"line\":" << line << ",\"column\":" << column
         << ",\"macroname\":\"" << name << "\",\"macrostmt\":\"" << stmt
         << "\"}";
      printf("%s\n", ss.str().c_str());
    }
  }

  /**
   * @brief 宏扩展处理回调函数
   *
   * 当源码中遇到宏扩展时，此函数被调用。函数主要用于检测宏所在位置是否合法（非系统头文件内），
   * 并记录宏扩展的信息（包括文件路径、行号、列号、宏名称及其原始声明）。记录的信息会以
   * JSON 格式输出，
   * 前缀的井号数量根据文件路径和内容匹配的结果而定，从而便于后续日志分析和调试处理。
   *
   * @param MacroNameTok 宏名称对应的标记对象，用于获取宏名称及其位置
   * @param MacroDefinition 当前宏的宏定义信息
   * @param Range 宏调用在源码中的范围
   * @param Args （可选）宏的参数列表
   */
  void MacroExpands(const Token &MacroNameTok,
                    const MacroDefinition &MacroDefinition, SourceRange Range,
                    const MacroArgs *Args) override {
    if (option.find("MacroDefExpIfndefInclusionCommentSkip|") ==
        std::string::npos)
      return;
    SourceManager &SM = PP.getSourceManager();
    SourceLocation Loc = MacroNameTok.getLocation();
    if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
      return;
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    std::string file = getPath(PLoc.getFilename());
    int line = PLoc.getLine();
    int column = PLoc.getColumn();
    std::string raw_stmt = MacroNameTok.getIdentifierInfo()->getName().str();
    std::string name = escapeJsonString(raw_stmt);
    std::string stmt = name;
    std::stringstream ss;
    if (skipPath(file)) {
      ss << "##";
    } else if (!matchFileContent(file, line, column, raw_stmt)) {
      ss << "####";
    }
    ss << "{\"type\":\"MacroExpands\",\"file\":\"" << file
       << "\",\"line\":" << line << ",\"column\":" << column
       << ",\"macroname\":\"" << name << "\",\"macrostmt\":\"" << stmt << "\"}";
    printf("%s\n", ss.str().c_str());
  }

  /**
   * @brief 处理 #ifndef 指令，生成并输出宏定义相关的 JSON 注释信息。
   *
   * 此函数用于在预处理阶段对 #ifndef 指令进行分析，
   * 并输出包含文件路径、行号、列号、宏名称及宏语句的 JSON 格式的调试注释。
   *
   * 主要过程：
   * 1. 检查配置选项中是否包含
   * "MacroDefExpIfndefInclusionCommentSkip|"，如果不包含则跳过处理。
   * 2.
   * 获取当前宏定义的源码位置信息，若位置无效或位于系统头文件中，则不做处理直接返回。
   * 3. 根据宏名称 Token 获取文件名、行号和列号，并对宏名称进行 JSON 格式转义。
   * 4.
   * 判断文件路径是否应跳过（skipPath），或文件内容是否符合预期（matchFileContent）：
   *    - 如果文件路径满足跳过条件，则在输出 JSON 前添加 "##" 前缀；
   *    - 如果文件内容匹配失败，则添加 "####" 前缀。
   * 5. 最后，将这些信息格式化为 JSON 字符串，并使用 printf 输出。
   *
   * @param Loc         宏定义的源代码位置（SourceLocation）。
   * @param MacroNameTok 宏名称对应的 Token 对象。
   * @param MD          宏定义的详细信息。
   */
  void Ifndef(SourceLocation Loc, const Token &MacroNameTok,
              const MacroDefinition &MD) override {
    if (option.find("MacroDefExpIfndefInclusionCommentSkip|") ==
        std::string::npos)
      return;
    SourceManager &SM = PP.getSourceManager();
    if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
      return;
    PresumedLoc PLoc = SM.getPresumedLoc(MacroNameTok.getLocation());
    std::string file = getPath(PLoc.getFilename());
    int line = PLoc.getLine();
    int column = PLoc.getColumn();
    std::string raw_stmt = MacroNameTok.getIdentifierInfo()->getName().str();
    std::string name = escapeJsonString(raw_stmt);
    std::string stmt = name;
    std::stringstream ss;
    if (skipPath(file)) {
      ss << "##";
    } else if (!matchFileContent(file, line, column, raw_stmt)) {
      ss << "####";
    }
    ss << "{\"type\":\"Ifndef\",\"file\":\"" << file << "\",\"line\":" << line
       << ",\"column\":" << column << ",\"macroname\":\"" << name
       << "\",\"macrostmt\":\"" << stmt << "\"}";
    printf("%s\n", ss.str().c_str());
  }

  /**
   * @brief #include 处理函数
   *
   * 当预处理器遇到 #incliude 指令时，会调用此函数处理相关信息，
   * 如指令的位置、包含文件的名称、是否使用尖括号形式包含、
   * 文件名在源文件中的范围、相关搜索路径、以及导入的模块信息等。
   *
   * 参数说明：
   *   @param HashLoc '#' 符号的位置
   *   @param IncludeTok 包含指令的标记对象
   *   @param FileName 被包含文件的名称
   *   @param IsAngled 指示是否使用尖括号 (<>) 包含文件
   *   @param FilenameRange 文件名在源码中的字符范围
   *   @param File 可选的文件实体引用
   *   @param SearchPath 搜索路径字符串
   *   @param RelativePath 文件相对于搜索路径的相对路径
   *   @param Imported 导入的模块指针
   *   @param FileType 文件的特性类型
   */
  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange,
                          OptionalFileEntryRef File, StringRef SearchPath,
                          StringRef RelativePath, const Module *Imported,
                          SrcMgr::CharacteristicKind FileType) override {
    if (option.find("MacroDefExpIfndefInclusionCommentSkip|") ==
        std::string::npos)
      return;
    SourceManager &SM = PP.getSourceManager();
    SourceLocation Loc = SM.getSpellingLoc(IncludeTok.getLocation());
    if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
      return;
    PresumedLoc PLoc = SM.getPresumedLoc(FilenameRange.getBegin());
    std::string file = getPath(PLoc.getFilename());
    int line = PLoc.getLine();
    int column = PLoc.getColumn();
    std::string includeFilePath{getPath(File->getName().str())};
    SourceLocation BeginLoc = FilenameRange.getBegin();
    SourceLocation EndLoc = Lexer::getLocForEndOfToken(FilenameRange.getEnd(),
                                                       0, SM, LangOptions());
    CharSourceRange FullRange = CharSourceRange::getCharRange(BeginLoc, EndLoc);
    std::string raw_stmt =
        Lexer::getSourceText(FullRange, SM, LangOptions()).str();
    std::string stmt = escapeJsonString(raw_stmt);
    std::stringstream ss;
    if (skipPath(file)) {
      ss << "##";
    } else if (skipPath(includeFilePath)) {
      ss << "##";
    } else if (!matchFileContent(file, line, column, raw_stmt)) {
      ss << "####";
    }
    ss << "{\"type\":\"InclusionDirective\",\"file\":\"" << file
       << "\",\"line\":" << line << ",\"column\":" << column
       << ",\"includefilepath\":\"" << includeFilePath << "\",\"stmt\":\""
       << stmt << "\"}";
    printf("%s\n", ss.str().c_str());
  }

  /**
   * @brief 当源代码范围被跳过时的回调处理函数
   *
   * 此函数用于处理在预处理阶段跳过的代码区域。当检测到选项字符串中包含特定标识符后，
   * 它会获取源代码的起始位置，并进一步通过源管理器判断该位置是否合法以及是否位于系统头文件中。
   * 如果位置无效或在系统头文件中，则函数直接返回，不进行后续处理。
   *
   * 函数随后通过源位置获取源代码中的完整字符串，并对该字符串进行 JSON
   * 转义处理。 然后，函数会遍历预定义的宏定义集合
   * MD_vec，若当前文件和语句内容与集合中某一项匹配，
   * 则直接返回。否则，依据文件路径过滤条件（通过 skipPath
   * 判断）或内容匹配检查（matchFileContent） 的结果，构造不同前缀的 JSON
   * 格式消息，并将包含类型、文件路径、行号、列号、以及经过转义的语句 信息的
   * JSON 字符串输出。
   *
   * @param Range    表示需要检查的源代码范围
   * @param EndifLoc 表示结束条件编译块的位置（当前未直接使用）
   */
  void SourceRangeSkipped(SourceRange Range, SourceLocation EndifLoc) override {
    if (option.find("MacroDefExpIfndefInclusionCommentSkip|") ==
        std::string::npos)
      return;
    SourceManager &SM = PP.getSourceManager();
    SourceLocation Loc = SM.getSpellingLoc(Range.getBegin());
    if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
      return;
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    std::string file = getPath(PLoc.getFilename());
    int line = PLoc.getLine();
    int column = PLoc.getColumn();
    CharSourceRange FullRange =
        CharSourceRange::getCharRange(Range.getBegin(), Range.getEnd());
    std::string raw_stmt =
        Lexer::getSourceText(FullRange, SM, LangOptions()).str();
    std::string stmt = escapeJsonString(raw_stmt);
    for (const auto &MD : MD_vec) {
      if (file == MD.file && raw_stmt.find(MD.stmt) != std::string::npos) {
        return;
      }
    }
    std::stringstream ss;
    if (skipPath(file)) {
      ss << "##";
    } else if (!matchFileContent(file, line, column, raw_stmt)) {
      ss << "####";
    }
    ss << "{\"type\":\"SourceRangeSkipped\",\"file\":\"" << file
       << "\",\"line\":" << line << ",\"column\":" << column << ",\"stmt\":\""
       << stmt << "\"}";
    printf("%s\n", ss.str().c_str());
  }
};

class AnalysisComment : public CommentHandler {
public:
  // 处理注释的回调函数
  // 当预处理器检测到注释时，将调用此函数进行处理
  // 参数说明：
  //   PP: 当前预处理器实例的引用
  //   Comment: 表示注释所在的源代码范围
  // 返回值：
  //   如果成功处理注释则返回 true，否则返回 false
  bool HandleComment(Preprocessor &PP, SourceRange Comment) override {
    if (option.find("MacroDefExpIfndefInclusionCommentSkip|") ==
        std::string::npos)
      return false;
    SourceManager &SM = PP.getSourceManager();
    SourceLocation Loc = SM.getSpellingLoc(Comment.getBegin());
    if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
      return false;
    PresumedLoc PLoc = SM.getPresumedLoc(Comment.getBegin());
    std::string file = getPath(PLoc.getFilename());
    int line = PLoc.getLine();
    int column = PLoc.getColumn();
    SourceLocation BeginLoc = Comment.getBegin();
    SourceLocation EndLoc = Comment.getEnd();
    CharSourceRange FullRange = CharSourceRange::getCharRange(BeginLoc, EndLoc);
    std::string raw_stmt =
        Lexer::getSourceText(FullRange, SM, LangOptions()).str();
    std::string stmt = escapeJsonString(raw_stmt);
    std::stringstream ss;
    if (skipPath(file)) {
      ss << "##";
    } else if (!matchFileContent(file, line, column, raw_stmt)) {
      ss << "####";
    }
    ss << "{\"type\":\"Comment\",\"file\":\"" << file << "\",\"line\":" << line
       << ",\"column\":" << column << ",\"stmt\":\"" << stmt << "\"}";
    printf("%s\n", ss.str().c_str());
    return false;
  }
};

class PPAnalysisAction : public PreprocessorFrontendAction {
protected:
  // 重写 ExecuteAction 函数，执行相应的预处理动作
  void ExecuteAction() override {
    Preprocessor &PP = getCompilerInstance().getPreprocessor();
    PP.addPPCallbacks(std::make_unique<AnalysisPPCallback>(PP));
    PP.addCommentHandler(new AnalysisComment);
    PP.EnterMainSourceFile();
    Token Tok;
    do {
      PP.Lex(Tok);
    } while (Tok.isNot(tok::eof));
  }
};

class AnalysisMatchCallback : public MatchFinder::MatchCallback {
public:
  explicit AnalysisMatchCallback(ASTContext &Context, SourceManager &SM)
      : Context(Context), SM(SM) {}

  /**
   * @brief 运行分析器，根据 AST 匹配结果执行相应的处理操作输出 JSON 信息。
   *
   * 此函数接收一个 MatchFinder::MatchResult 对象，根据其中匹配的 AST 节点类型，
   * 执行以下处理：
   *   - 如果匹配到 "SkipNamespaceDecl" 的 NamespaceDecl
   * 节点，检查其源位置和文件路径，
   *     并在满足条件时输出包含文件名、行号、列号以及命名空间名称的 JSON
   * 信息，同时 将该命名空间添加到忽略集合中。
   *
   *   - 如果匹配到 "NamespaceDecl" 的 NamespaceDecl
   * 节点，提取源文件路径、位置和名称， 检查文件是否需要跳过，并生成相应的 JSON
   * 信息。
   *
   *   - 如果匹配到 DeclRefExpr
   * 节点，提取表达式的源文本、其引用的声明位置等信息，
   *     并根据是否处于宏定义或者文件内容校验，输出处理前缀（如 "##",
   * "####"）以及 包含类型信息和对应声明位置的 JSON 信息。
   *
   *   - 如果匹配到 TypeLoc 节点，根据类型种类（例如
   * Record、Typedef、Enum、TemplateSpecialization 等）
   *     执行相应处理，提取类型名称、所属声明的位置，并输出 JSON
   * 信息；对指针、引用、内建类型等情况下， 输出特殊标识前缀。
   *
   *   - 如果匹配到 FunctionDecl
   * 节点，则提取函数声明或定义的源码范围，若存在函数体则只截取函数
   *     定义前部分的源码作为名称，结合返回类型生成最终输出的 JSON 信息。
   *
   *   - 如果匹配到 UsingDirectiveDecl
   * 节点，则提取命名空间使用指令相关的信息，检查文件跳过条件，
   *     并输出包含使用指令源文本及目标命名空间位置的 JSON 信息。
   *
   *   - 如果匹配到 NamedDecl
   * 节点，根据声明类型和名称，检查文件跳过条件后输出相应的 JSON 信息，
   *     同时对部分类型（如 Namespace、Using、Function 等）添加特殊前缀标识。
   *
   *   - 如果匹配到 MemberExpr
   * 节点，提取成员名称及其声明的位置，检查文件条件后生成 JSON 信息。
   *
   *   - 如果匹配到 CXXCtorInitializer
   * 节点（仅处理成员初始化器），提取成员名称和对应的声明位置，
   *     并在满足输出条件时生成 JSON 信息。
   *
   *   - 如果匹配到 CallExpr
   * 节点，提取函数调用的源文本并获取被调用函数的名称及其声明位置，
   *     结合文件路径检查和内容匹配结果输出最终 JSON 信息。
   *
   * 在每个分支中，函数首先检查源位置的有效性、是否在系统头文件中以及是否符合跳过条件，
   * 然后对提取的源文本进行 JSON 字符串转义和路径过滤，最后将处理结果以 JSON
   * 格式打印输出。
   *
   * @param Result 存放 AST 匹配结果的对象，用于传递需要处理的语法节点信息。
   */
  virtual void run(const MatchFinder::MatchResult &Result) {
    if (const NamespaceDecl *ND =
            Result.Nodes.getNodeAs<NamespaceDecl>("SkipNamespaceDecl|")) {
      std::string file, Dfile, raw_stmt, stmt, Dstmt;
      int line = 0, column = 0;
      int Dline = 0, Dcolumn = 0;
      static std::set<std::string> skipNamespaceDeclSet;
      SourceLocation Loc = SM.getSpellingLoc(ND->getLocation());
      if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
        return;
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      if (PLoc.isValid()) {
        file = getPath(PLoc.getFilename());
        line = PLoc.getLine();
        column = PLoc.getColumn();
        raw_stmt = ND->getNameAsString();
        stmt = escapeJsonString(raw_stmt);
      }
      std::stringstream ss;
      if (stmt.size() == 0 || skipNamespaceDeclSet.count(stmt) > 0) {
        return;
      } else if (skipPath(file)) {
        skipNamespaceDeclSet.insert(stmt);
        ss << "{\"type\":\"SkipNamespaceDecl\",\"file\":\"" << file
           << "\",\"line\":" << line << ",\"column\":" << column
           << ",\"stmt\":\"" << stmt << "\"}";
        printf("%s\n", ss.str().c_str());
      }
    } else if (const NamespaceDecl *ND =
                   Result.Nodes.getNodeAs<NamespaceDecl>("NamespaceDecl|")) {
      std::string file, Dfile, raw_stmt, stmt, Dstmt;
      int line = 0, column = 0;
      int Dline = 0, Dcolumn = 0;
      SourceLocation Loc = SM.getSpellingLoc(ND->getLocation());
      if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
        return;
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      if (PLoc.isValid()) {
        file = getPath(PLoc.getFilename());
        line = PLoc.getLine();
        column = PLoc.getColumn();
        raw_stmt = ND->getNameAsString();
        stmt = escapeJsonString(raw_stmt);
      }
      std::stringstream ss;
      if (skipPath(file)) {
        return;
      } else if (!matchFileContent(file, line, column, raw_stmt)) {
        ss << "####";
      }
      ss << "{\"type\":\"NamespaceDecl\",\"file\":\"" << file
         << "\",\"line\":" << line << ",\"column\":" << column << ",\"stmt\":\""
         << stmt << "\",\"exptype\":\""
         << "NamespaceDecl"
         << "\",\"dfile\":\"" << Dfile << "\",\"dline\":" << Dline
         << ",\"dcolumn\":" << Dcolumn << ",\"dstmt\":\"" << Dstmt << "\"}";
      printf("%s\n", ss.str().c_str());
    } else if (const DeclRefExpr *DRE =
                   Result.Nodes.getNodeAs<DeclRefExpr>("DeclRefExpr|")) {
      std::string file, Dfile, raw_stmt, stmt, raw_Dstmt, Dstmt;
      int line = 0, column = 0;
      int Dline = 0, Dcolumn = 0;
      SourceLocation Loc = SM.getSpellingLoc(DRE->getBeginLoc());
      bool checkLoc = SM.isWrittenInSameFile(Loc, DRE->getLocation());
      if (!checkLoc) {
        Loc = SM.getSpellingLoc(DRE->getLocation());
      }
      if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
        return;
      std::string exptype = DRE->getType()->getTypeClassName();
      {
        CharSourceRange FullRange;
        if (DRE->getBeginLoc().isMacroID()) {
          if (checkLoc)
            FullRange = CharSourceRange::getTokenRange(
                SM.getSpellingLoc(DRE->getBeginLoc()),
                SM.getSpellingLoc(DRE->getEndLoc()));
          else {
            auto nextToken =
                Lexer::findNextToken(DRE->getLocation(), SM, LangOptions());
            FullRange = CharSourceRange::getCharRange(
                SM.getSpellingLoc(DRE->getLocation()),
                SM.getSpellingLoc(nextToken->getLocation()));
          }
        } else {
          if (checkLoc)
            FullRange = CharSourceRange::getTokenRange(DRE->getBeginLoc(),
                                                       DRE->getEndLoc());
          else {
            auto nextToken =
                Lexer::findNextToken(DRE->getLocation(), SM, LangOptions());
            FullRange = CharSourceRange::getCharRange(DRE->getLocation(),
                                                      nextToken->getLocation());
          }
        }
        raw_stmt = Lexer::getSourceText(FullRange, SM, LangOptions()).str();
        stmt = escapeJsonString(raw_stmt);
      }
      const ValueDecl *D = DRE->getDecl();
      SourceLocation DLoc = SM.getSpellingLoc(D->getBeginLoc());
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      if (PLoc.isValid()) {
        file = getPath(PLoc.getFilename());
        line = PLoc.getLine();
        column = PLoc.getColumn();
      }
      if (DLoc.isInvalid() || SM.isInSystemHeader(DLoc))
        return;
      PresumedLoc DPLoc = SM.getPresumedLoc(DLoc);
      if (DPLoc.isValid()) {
        Dfile = getPath(DPLoc.getFilename());
        Dline = DPLoc.getLine();
        Dcolumn = DPLoc.getColumn();
      }
      Dstmt = "DeclRefExpr";
      std::string macro;
      if (option.find("MacroDefExpIfndefInclusionCommentSkip|") !=
          std::string::npos) {
        macro = isInsideMacroDefinition(file, line, column);
        if (macro.empty())
          return;
      }
      std::stringstream ss;
      if (!macro.empty()) {
        CharSourceRange FullRange = CharSourceRange::getTokenRange(
            SM.getSpellingLoc(DRE->getBeginLoc()),
            SM.getSpellingLoc(DRE->getEndLoc()));
        raw_stmt = Lexer::getSourceText(FullRange, SM, LangOptions()).str();
        stmt = escapeJsonString(raw_stmt);
        if (!Dfile.empty() && skipPath(Dfile)) {
          raw_stmt += ("%%!" + macro);
        } else {
          raw_stmt += ("%%" + macro);
        }
        stmt = escapeJsonString(raw_stmt);
      } else if (skipPath(file)) {
        ss << "##";
      } else if (!Dfile.empty() && skipPath(Dfile)) {
        ss << "##-";
      } else if (!matchFileContent(file, line, column, raw_stmt)) {
        ss << "####";
      }
      ss << "{\"type\":\"DeclRefExprTypeLoc\",\"file\":\"" << file
         << "\",\"line\":" << line << ",\"column\":" << column << ",\"stmt\":\""
         << stmt << "\",\"exptype\":\"" << exptype << "\",\"dfile\":\"" << Dfile
         << "\",\"dline\":" << Dline << ",\"dcolumn\":" << Dcolumn
         << ",\"dstmt\":\"" << Dstmt << "\"}";
      printf("%s\n", ss.str().c_str());
    } else if (const TypeLoc *TL =
                   Result.Nodes.getNodeAs<TypeLoc>("TypeLoc|")) {
      std::string file, Dfile, raw_stmt, stmt, Dstmt;
      int line = 0, column = 0;
      int Dline = 0, Dcolumn = 0;
      static int prev_line = 0, prev_column = 0;
      SourceLocation Loc = SM.getSpellingLoc(TL->getBeginLoc());
      if (Loc.isInvalid() || SM.isInSystemHeader(Loc)) {
        return;
      }
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      if (PLoc.isValid()) {
        file = getPath(PLoc.getFilename());
        line = PLoc.getLine();
        column = PLoc.getColumn();
      }
      {
        CharSourceRange FullRange;
        if (TL->getBeginLoc().isMacroID()) {
          FullRange = CharSourceRange::getCharRange(
              SM.getSpellingLoc(TL->getBeginLoc()),
              SM.getSpellingLoc(TL->getEndLoc()));
        } else {
          FullRange =
              CharSourceRange::getCharRange(TL->getBeginLoc(), TL->getEndLoc());
        }
        raw_stmt = Lexer::getSourceText(FullRange, SM, LangOptions()).str();
        stmt = escapeJsonString(raw_stmt);
      }
      SourceLocation DLoc = Loc;
      std::stringstream ss;
      switch (TL->getType()->getTypeClass()) {
      case Type::TypeClass::Record:
        if (RecordTypeLoc RTL = TL->getAs<RecordTypeLoc>()) {
          RecordDecl *D = RTL.getDecl();
          DLoc = D->getBeginLoc();
          raw_stmt = D->getNameAsString();
          std::string Scope;
          const DeclContext *DC = D->getDeclContext();
          while (DC && isa<NamedDecl>(DC)) {
            if (isa<NamespaceDecl>(DC))
              Scope = dyn_cast<NamedDecl>(DC)->getNameAsString();
            else if (isa<CXXRecordDecl>(DC))
              Scope = dyn_cast<NamedDecl>(DC)->getNameAsString();
            else
              break;
            DC = DC->getParent();
          }
          int Scope_line = 0, Scope_column = 0;
          Scope = Scope.empty() ? ""
                                : getStmtBefore(file, line, column,
                                                Scope + "\\s*::", &Scope_line,
                                                &Scope_column);
          if (Scope_line > 0 && Scope_column > 0) {
            if (Scope_line > prev_line ||
                (Scope_line == prev_line && Scope_column >= prev_column)) {
              line = Scope_line;
              column = Scope_column;
              raw_stmt = Scope + raw_stmt;
            }
          }
          stmt = escapeJsonString(raw_stmt);
        }
        break;
      case Type::TypeClass::Typedef:
        if (TypedefTypeLoc TTL = TL->getAs<TypedefTypeLoc>()) {
          TypedefNameDecl *TND = TTL.getTypedefNameDecl();
          DLoc = TND->getBeginLoc();
          raw_stmt = TND->getNameAsString();
          stmt = escapeJsonString(raw_stmt);
        }
        break;
      case Type::TypeClass::Enum:
        if (EnumTypeLoc ETL = TL->getAs<EnumTypeLoc>()) {
          EnumDecl *D = ETL.getDecl();
          DLoc = D->getBeginLoc();
          raw_stmt = D->getNameAsString();
          stmt = escapeJsonString(raw_stmt);
        }
        break;
      case Type::TypeClass::TemplateSpecialization:
        if (TemplateSpecializationTypeLoc TSTL =
                TL->getAs<TemplateSpecializationTypeLoc>()) {
          if (const TemplateSpecializationType *TST =
                  TSTL.getTypePtr()->getAs<TemplateSpecializationType>()) {
            if (TemplateDecl *TD = TST->getTemplateName().getAsTemplateDecl()) {
              DLoc = TD->getLocation();
            }
          }
        }
        break;
      case Type::TypeClass::Pointer:
      case Type::TypeClass::LValueReference:
      case Type::TypeClass::RValueReference:
      case Type::TypeClass::Builtin:
      case Type::TypeClass::FunctionProto:
        ss << "##";
        break;
      default:
        break;
      }
      if (DLoc.isInvalid() || SM.isInSystemHeader(DLoc))
        return;
      PresumedLoc DPLoc = SM.getPresumedLoc(DLoc);
      if (DPLoc.isValid()) {
        Dfile = getPath(DPLoc.getFilename());
        Dline = DPLoc.getLine();
        Dcolumn = DPLoc.getColumn();
      }
      Dstmt = "TypeLoc";
      QualType QT = TL->getType();
      std::string exptype = QT->getTypeClassName();
      if (skipPath(file)) {
        ss << "##";
      } else if (!Dfile.empty() && skipPath(Dfile)) {
        ss << "##-";
      } else if (!matchFileContent(file, line, column, raw_stmt)) {
        ss << "####";
      }
      ss << "{\"type\":\"DeclRefExprTypeLoc\",\"file\":\"" << file
         << "\",\"line\":" << line << ",\"column\":" << column << ",\"stmt\":\""
         << stmt << "\",\"exptype\":\"" << exptype << "\",\"dfile\":\"" << Dfile
         << "\",\"dline\":" << Dline << ",\"dcolumn\":" << Dcolumn
         << ",\"dstmt\":\"" << Dstmt << "\"}";
      std::string content = ss.str();
      printf("%s\n", content.c_str());
      {
        PresumedLoc EPLoc = SM.getPresumedLoc(TL->getSourceRange().getEnd());
        prev_line = EPLoc.getLine();
        prev_column = EPLoc.getColumn();
      }
    } else if (const FunctionDecl *FD =
                   Result.Nodes.getNodeAs<FunctionDecl>("FunctionDecl|")) {
      std::string file, Dfile, raw_stmt, stmt, Dstmt;
      int line = 0, column = 0;
      int Dline = 0, Dcolumn = 0;
      SourceLocation Loc = SM.getSpellingLoc(FD->getBeginLoc());
      if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
        return;
      const CompoundStmt *Body = dyn_cast_or_null<CompoundStmt>(FD->getBody());
      if (Body != nullptr) {
        CharSourceRange FullRange = CharSourceRange::getTokenRange(
            FD->getBeginLoc(), FD->getBody()->getBeginLoc());
        raw_stmt = Lexer::getSourceText(FullRange, SM, LangOptions()).str();
        stmt = escapeJsonString(raw_stmt);
      } else {
        CharSourceRange FullRange =
            CharSourceRange::getTokenRange(FD->getBeginLoc(), FD->getEndLoc());
        raw_stmt = Lexer::getSourceText(FullRange, SM, LangOptions()).str();
        stmt = escapeJsonString(raw_stmt);
      }
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      if (PLoc.isValid()) {
        file = getPath(PLoc.getFilename());
        line = PLoc.getLine();
        column = PLoc.getColumn();
      }
      SourceRange SR = FD->getReturnTypeSourceRange();
      std::string Rstmt = Lexer::getSourceText(CharSourceRange::getTokenRange(
                                                   SR.getBegin(), SR.getEnd()),
                                               SM, LangOptions())
                              .str();
      if (!stmt.empty() && !Rstmt.empty())
        stmt = escapeJsonString(raw_stmt + "%%" + Rstmt);
      std::stringstream ss;
      if (skipPath(file)) {
        ss << "##-";
      } else if (!matchFileContent(file, line, column, raw_stmt)) {
        ss << "####";
      }
      ss << "{\"type\":\"FunctionDecl\",\"file\":\"" << file
         << "\",\"line\":" << line << ",\"column\":" << column << ",\"stmt\":\""
         << stmt << "\",\"exptype\":\""
         << (!FD->isThisDeclarationADefinition() || FD->getBody() == nullptr
                 ? "FunctionDecl"
                 : "FunctionImpl")
         << "\",\"dfile\":\"" << Dfile << "\",\"dline\":" << Dline
         << ",\"dcolumn\":" << Dcolumn << ",\"dstmt\":\"" << Dstmt << "\"}";
      printf("%s\n", ss.str().c_str());
    } else if (const UsingDirectiveDecl *UDD =
                   Result.Nodes.getNodeAs<UsingDirectiveDecl>(
                       "UsingDirectiveDecl|")) {
      std::string file, Dfile, raw_stmt, stmt, Dstmt;
      int line = 0, column = 0;
      int Dline = 0, Dcolumn = 0;
      SourceLocation Loc = SM.getSpellingLoc(UDD->getBeginLoc());
      if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
        return;
      {
        SourceLocation End = Lexer::findLocationAfterToken(
            UDD->getEndLoc(), clang::tok::semi, SM, LangOptions(), false);
        CharSourceRange FullRange =
            CharSourceRange::getCharRange(UDD->getBeginLoc(), End);
        raw_stmt = Lexer::getSourceText(FullRange, SM, LangOptions()).str();
        stmt = escapeJsonString(escapeJsonString(raw_stmt));
      }
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      if (PLoc.isValid()) {
        file = getPath(PLoc.getFilename());
        line = PLoc.getLine();
        column = PLoc.getColumn();
      }
      SourceLocation DLoc = UDD->getNominatedNamespace()->getBeginLoc();
      if (DLoc.isInvalid() || SM.isInSystemHeader(DLoc))
        return;
      PresumedLoc DPLoc = SM.getPresumedLoc(DLoc);
      if (DPLoc.isValid()) {
        Dfile = getPath(DPLoc.getFilename());
        Dline = DPLoc.getLine();
        Dcolumn = DPLoc.getColumn();
      }
      std::stringstream ss;
      if (skipPath(file)) {
        ss << "##";
      } else if (!Dfile.empty() && skipPath(Dfile)) {
        ss << "##";
      } else if (!matchFileContent(file, line, column, raw_stmt)) {
        ss << "####";
      }
      ss << "{\"type\":\"UsingDirectiveDecl\",\"file\":\"" << file
         << "\",\"line\":" << line << ",\"column\":" << column << ",\"stmt\":\""
         << stmt << "\",\"exptype\":\""
         << "UsingDirectiveDecl"
         << "\",\"dfile\":\"" << Dfile << "\",\"dline\":" << Dline
         << ",\"dcolumn\":" << Dcolumn << ",\"dstmt\":\"" << Dstmt << "\"}";
      printf("%s\n", ss.str().c_str());
    } else if (const NamedDecl *ND =
                   Result.Nodes.getNodeAs<NamedDecl>("NamedDecl|")) {
      std::string file, Dfile, raw_stmt, stmt, Dstmt;
      int line = 0, column = 0;
      int Dline = 0, Dcolumn = 0;
      SourceLocation Loc = SM.getSpellingLoc(ND->getLocation());
      if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
        return;
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      if (PLoc.isValid()) {
        file = getPath(PLoc.getFilename());
        line = PLoc.getLine();
        column = PLoc.getColumn();
      }
      raw_stmt = ND->getNameAsString();
      stmt = escapeJsonString(raw_stmt);
      Dstmt = "NamedDecl";
      std::string exptype = ND->getDeclKindName();
      std::stringstream ss;
      switch (ND->getKind()) {
      case NamedDecl::Namespace:
      case NamedDecl::Using:
      case NamedDecl::UsingDirective:
      case NamedDecl::Function:
      case NamedDecl::FunctionTemplate:
      case NamedDecl::CXXMethod:
        ss << "##";
      default:
        break;
      }
      if (skipPath(file)) {
        ss << "##-";
      } else if (!matchFileContent(file, line, column, raw_stmt)) {
        ss << "####";
      }
      ss << "{\"type\":\"NamedDeclMemberExpr\",\"file\":\"" << file
         << "\",\"line\":" << line << ",\"column\":" << column << ",\"stmt\":\""
         << stmt << "\",\"exptype\":\"" << exptype << "\",\"dfile\":\"" << Dfile
         << "\",\"dline\":" << Dline << ",\"dcolumn\":" << Dcolumn
         << ",\"dstmt\":\"" << Dstmt << "\"}";
      printf("%s\n", ss.str().c_str());
    } else if (const MemberExpr *ME =
                   Result.Nodes.getNodeAs<MemberExpr>("MemberExpr|")) {
      std::string file, Dfile, raw_stmt, stmt, Dstmt;
      int line = 0, column = 0;
      int Dline = 0, Dcolumn = 0;
      SourceLocation Loc = SM.getSpellingLoc(ME->getBeginLoc());
      if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
        return;
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      if (PLoc.isValid()) {
        file = getPath(PLoc.getFilename());
        line = PLoc.getLine();
        column = PLoc.getColumn();
      }
      if (ValueDecl *VD = ME->getMemberDecl()) {
        SourceLocation DLoc = SM.getSpellingLoc(VD->getBeginLoc());
        if (DLoc.isInvalid() || SM.isInSystemHeader(DLoc))
          return;
        PresumedLoc DPLoc = SM.getPresumedLoc(DLoc);
        if (DPLoc.isValid()) {
          Dfile = getPath(DPLoc.getFilename());
          Dline = DPLoc.getLine();
          Dcolumn = DPLoc.getColumn();
        }
        DeclarationName DN = VD->getDeclName();
        raw_stmt = DN.getAsString();
        stmt = escapeJsonString(raw_stmt);
      }
      std::stringstream ss;
      if (skipPath(file)) {
        ss << "##";
      } else if (!Dfile.empty() && skipPath(Dfile)) {
        ss << "##";
      } else if (!matchFileContent(file, line, column, raw_stmt)) {
        ss << "####";
      }
      ss << "{\"type\":\"NamedDeclMemberExpr\",\"file\":\"" << file
         << "\",\"line\":" << line << ",\"column\":" << column << ",\"stmt\":\""
         << stmt << "\",\"exptype\":\""
         << "MemberExpr"
         << "\",\"dfile\":\"" << Dfile << "\",\"dline\":" << Dline
         << ",\"dcolumn\":" << Dcolumn << ",\"dstmt\":\"" << Dstmt << "\"}";
      printf("%s\n", ss.str().c_str());
    } else if (const CXXCtorInitializer *CI =
                   Result.Nodes.getNodeAs<CXXCtorInitializer>(
                       "CXXCtorInitializer|")) {
      if (!CI->isMemberInitializer())
        return;
      std::string file, Dfile, raw_stmt, stmt, Dstmt;
      int line = 0, column = 0;
      int Dline = 0, Dcolumn = 0;
      SourceLocation Loc = SM.getSpellingLoc(CI->getSourceLocation());
      if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
        return;
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      if (PLoc.isValid()) {
        file = getPath(PLoc.getFilename());
        line = PLoc.getLine();
        column = PLoc.getColumn();
      }
      if (FieldDecl *F = CI->getMember()) {
        SourceLocation DLoc = SM.getSpellingLoc(F->getBeginLoc());
        if (DLoc.isInvalid() || SM.isInSystemHeader(DLoc))
          return;
        PresumedLoc DPLoc = SM.getPresumedLoc(DLoc);
        if (DPLoc.isValid()) {
          Dfile = getPath(DPLoc.getFilename());
          Dline = DPLoc.getLine();
          Dcolumn = DPLoc.getColumn();
        }
        raw_stmt = F->getNameAsString();
        stmt = escapeJsonString(raw_stmt);
      }
      std::stringstream ss;
      if (skipPath(file)) {
        ss << "##";
      } else if (!Dfile.empty() && skipPath(Dfile)) {
        ss << "##";
      } else if (!matchFileContent(file, line, column, raw_stmt)) {
        ss << "####";
      }
      ss << "{\"type\":\"NamedDeclMemberExpr\",\"file\":\"" << file
         << "\",\"line\":" << line << ",\"column\":" << column << ",\"stmt\":\""
         << stmt << "\",\"exptype\":\""
         << "CXXCtorInitializer"
         << "\",\"dfile\":\"" << Dfile << "\",\"dline\":" << Dline
         << ",\"dcolumn\":" << Dcolumn << ",\"dstmt\":\"" << Dstmt << "\"}";
      printf("%s\n", ss.str().c_str());
    } else if (const CallExpr *CE =
                   Result.Nodes.getNodeAs<CallExpr>("CallExpr|")) {
      std::string file, Dfile, raw_stmt, stmt, Dstmt;
      int line = 0, column = 0;
      int Dline = 0, Dcolumn = 0;
      SourceLocation Loc = SM.getSpellingLoc(CE->getBeginLoc());
      if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
        return;
      {
        CharSourceRange FullRange =
            CharSourceRange::getTokenRange(CE->getBeginLoc(), CE->getEndLoc());
        raw_stmt = Lexer::getSourceText(FullRange, SM, LangOptions()).str();
        stmt = escapeJsonString(raw_stmt);
      }
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      if (PLoc.isValid()) {
        file = getPath(PLoc.getFilename());
        line = PLoc.getLine();
        column = PLoc.getColumn();
      }
      SourceLocation DLoc = Loc;
      if (const FunctionDecl *FD = CE->getDirectCallee()) {
        DLoc = FD->getBeginLoc();
        stmt = escapeJsonString(raw_stmt + "%%" + FD->getNameAsString());
      }
      if (DLoc.isInvalid() || SM.isInSystemHeader(DLoc))
        return;
      PresumedLoc DPLoc = SM.getPresumedLoc(DLoc);
      if (DPLoc.isValid()) {
        Dfile = getPath(DPLoc.getFilename());
        Dline = DPLoc.getLine();
        Dcolumn = DPLoc.getColumn();
      }
      std::stringstream ss;
      if (skipPath(file)) {
        ss << "##";
      } else if (!Dfile.empty() && skipPath(Dfile)) {
        ss << "##";
      } else if (!matchFileContent(file, line, column, raw_stmt)) {
        ss << "####";
      }
      ss << "{\"type\":\"CallExpr\",\"file\":\"" << file
         << "\",\"line\":" << line << ",\"column\":" << column << ",\"stmt\":\""
         << stmt << "\",\"exptype\":\""
         << "CallExpr"
         << "\",\"dfile\":\"" << Dfile << "\",\"dline\":" << Dline
         << ",\"dcolumn\":" << Dcolumn << ",\"dstmt\":\"" << Dstmt << "\"}";
      printf("%s\n", ss.str().c_str());
    }
  }

private:
  ASTContext &Context;
  SourceManager &SM;
};

class AnalysisConsumer : public ASTConsumer {
public:
  explicit AnalysisConsumer(MatchFinder &Finder) : Finder(Finder) {}

  void HandleTranslationUnit(ASTContext &Context) override {
    Finder.matchAST(Context);
  }

private:
  MatchFinder Finder;
};

class AnalysisAction : public ASTFrontendAction {
public:
  /**
   * @brief 创建 AST 消费者，根据不同的匹配器选项配置 AST 匹配器。
   *
   * 此函数根据传入的编译器实例 CI 中获取的 AST 上下文和源管理器，创建一个
   * AnalysisMatchCallback 回调。 然后，根据全局或成员变量 option
   * 中包含的不同关键字，配置相应的 clang::ast_matchers 匹配规则，如：
   *   - "SkipNamespaceDecl|": 为 namespaceDecl
   * 添加匹配规则，用于跳过命名空间声明。
   *   - "NamespaceDecl|": 为 namespaceDecl 添加匹配规则，用于处理命名空间声明。
   *   - "MacroDefExpIfndefInclusionCommentSkip|": 为 declRefExpr
   * 添加匹配规则，处理与预处理器宏定义相关的引用表达式。
   *   - "DeclRefExprTypeLoc|": 同时为 declRefExpr 和 typeLoc
   * 添加匹配规则，处理声明引用表达式及其类型位置。
   *   - "FunctionDecl|": 为 functionDecl 添加匹配规则，处理函数声明。
   *   - "UsingDirectiveDecl|": 为 usingDirectiveDecl 添加匹配规则，处理 using
   * 指令声明。
   *   - "NamedDeclMemberExpr|": 分别为 namedDecl、memberExpr 以及
   * cxxCtorInitializer 添加匹配规则，
   *       处理命名声明、成员表达式和构造函数初始化。
   *   - "CallExpr|": 为 callExpr 添加匹配规则，处理函数调用表达式。
   *
   * 最后，函数返回一个基于 AnalysisConsumer 的智能指针，用于进一步消费 AST
   * 遍历结果。
   *
   * @param CI 编译器实例，用于获取 AST 上下文和源管理器。
   * @param file 当前处理的文件路径（字符串视图）。
   * @return std::unique_ptr<ASTConsumer> 指向创建的 AnalysisConsumer
   * 对象的智能指针。
   */
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef file) override {
    MatchFinder Finder;
    AnalysisMatchCallback *Callback{
        new AnalysisMatchCallback{CI.getASTContext(), CI.getSourceManager()}};
    if (option.find("SkipNamespaceDecl|") != std::string::npos) {
      Finder.addMatcher(
          clang::ast_matchers::namespaceDecl().bind("SkipNamespaceDecl|"),
          Callback);
    } else if (option.find("NamespaceDecl|") != std::string::npos) {
      Finder.addMatcher(
          clang::ast_matchers::namespaceDecl().bind("NamespaceDecl|"),
          Callback);
    } else if (option.find("MacroDefExpIfndefInclusionCommentSkip|") !=
               std::string::npos) {
      Finder.addMatcher(clang::ast_matchers::declRefExpr().bind("DeclRefExpr|"),
                        Callback);
    } else if (option.find("DeclRefExprTypeLoc|") != std::string::npos) {
      Finder.addMatcher(clang::ast_matchers::declRefExpr().bind("DeclRefExpr|"),
                        Callback);
      Finder.addMatcher(clang::ast_matchers::typeLoc().bind("TypeLoc|"),
                        Callback);
    } else if (option.find("FunctionDecl|") != std::string::npos) {
      Finder.addMatcher(
          clang::ast_matchers::functionDecl().bind("FunctionDecl|"), Callback);
    } else if (option.find("UsingDirectiveDecl|") != std::string::npos) {
      Finder.addMatcher(
          clang::ast_matchers::usingDirectiveDecl().bind("UsingDirectiveDecl|"),
          Callback);
    } else if (option.find("NamedDeclMemberExpr|") != std::string::npos) {
      Finder.addMatcher(clang::ast_matchers::namedDecl().bind("NamedDecl|"),
                        Callback);
      Finder.addMatcher(clang::ast_matchers::memberExpr().bind("MemberExpr|"),
                        Callback);
      Finder.addMatcher(
          clang::ast_matchers::cxxCtorInitializer().bind("CXXCtorInitializer|"),
          Callback);
    } else if (option.find("CallExpr|") != std::string::npos) {
      Finder.addMatcher(clang::ast_matchers::callExpr().bind("CallExpr|"),
                        Callback);
    }
    return std::make_unique<AnalysisConsumer>(Finder);
  }
};

static llvm::cl::OptionCategory Category("Analysis Options");
int main(int argc, const char **argv) {
  if (getenv("ANALYZE_CMD") != nullptr) {
    option = getenv("ANALYZE_CMD");
  }
  if (getenv("ANALYZE_SKIP_PATH")) {
    std::string skipPath = getenv("ANALYZE_SKIP_PATH");
    size_t start = 0;
    size_t end = skipPath.find(':');
    while (end != std::string::npos) {
      std::string token = skipPath.substr(start, end - start);
      if (!token.empty())
        skipPaths.emplace_back(token);
      start = end + 1;
      end = skipPath.find(':', start);
    }
    std::string token = skipPath.substr(start);
    if (!token.empty())
      skipPaths.emplace_back(token);
  }
  auto ExpectedParser = CommonOptionsParser::create(argc, argv, Category);
  if (!ExpectedParser) {
    llvm::errs() << llvm::toString(ExpectedParser.takeError());
    return 1;
  }
  ClangTool Tool(ExpectedParser->getCompilations(),
                 ExpectedParser->getSourcePathList());
  Tool.run(newFrontendActionFactory<PPAnalysisAction>().get());
  Tool.run(newFrontendActionFactory<AnalysisAction>().get());
  return 0;
}
