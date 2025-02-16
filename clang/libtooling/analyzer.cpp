#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/MacroArgs.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/FrontendAction.h>
#include <cstdlib>
#include <sstream>

using namespace clang;
using namespace clang::tooling;
using namespace llvm;

class AnalysisCallback : public PPCallbacks {
  Preprocessor &PP;

public:
  AnalysisCallback(Preprocessor &PP) : PP(PP) {}

  void MacroDefined(const Token &MacroNameTok,
                    const MacroDirective *MD) override {
    const MacroInfo *MI = MD->getMacroInfo();
    SourceManager &SM = PP.getSourceManager();
    SourceLocation Loc = SM.getSpellingLoc(MacroNameTok.getLocation());
    if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
      return;
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    std::string filename{PLoc.getFilename() != nullptr ? PLoc.getFilename()
                                                       : ""};
    if (filename.empty() || filename.find("<") != std::string::npos)
      return;
    SourceLocation BeginLoc =
        Lexer::GetBeginningOfToken(MI->getDefinitionLoc(), SM, LangOptions());
    SourceLocation EndLoc = Lexer::getLocForEndOfToken(
        MI->getDefinitionEndLoc(), 0, SM, LangOptions());
    CharSourceRange FullRange = CharSourceRange::getCharRange(BeginLoc, EndLoc);
    StringRef Text = Lexer::getSourceText(FullRange, SM, LangOptions());
    std::stringstream ss;
    ss << "{\"type\":\"macrodefine\",\"file\":\"" << PLoc.getFilename()
       << "\",\"line\":" << PLoc.getLine() << ",\"column\":" << PLoc.getColumn()
       << ",\"macroname\":\""
       << MacroNameTok.getIdentifierInfo()->getName().str() << "\",\"stmt\":\""
       << Text.str() << "\"}";
    printf("%s\n", ss.str().c_str());
  }

  void MacroExpands(const Token &MacroNameTok,
                    const MacroDefinition &MacroDefinition, SourceRange Range,
                    const MacroArgs *Args) override {
    SourceManager &SM = PP.getSourceManager();
    SourceLocation Loc = SM.getSpellingLoc(MacroNameTok.getLocation());
    if (Loc.isInvalid() || SM.isInSystemHeader(Loc) ||
        !SM.isWrittenInMainFile(Loc))
      return;
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    SourceLocation BeginLoc = Lexer::GetBeginningOfToken(
        MacroNameTok.getLocation(), SM, LangOptions());
    SourceLocation EndLoc =
        Lexer::getLocForEndOfToken(Range.getEnd(), 0, SM, LangOptions());
    CharSourceRange FullRange = CharSourceRange::getCharRange(BeginLoc, EndLoc);
    StringRef Text = Lexer::getSourceText(FullRange, SM, LangOptions());
    std::stringstream ss;
    ss << "{\"type\":\"macroexpand\",\"file\":\"" << PLoc.getFilename()
       << "\",\"line\":" << PLoc.getLine() << ",\"column\":" << PLoc.getColumn()
       << ",\"macroname\":\""
       << MacroNameTok.getIdentifierInfo()->getName().str() << "\",\"stmt\":\""
       << Text.str() << "\"}";
    printf("%s\n", ss.str().c_str());
  }

  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange,
                          OptionalFileEntryRef File, StringRef SearchPath,
                          StringRef RelativePath, const Module *Imported,
                          SrcMgr::CharacteristicKind FileType) override {
    SourceManager &SM = PP.getSourceManager();
    SourceLocation Loc = SM.getSpellingLoc(IncludeTok.getLocation());
    if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
      return;
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    std::stringstream ss;
    std::string realIncludePath{
        realpath(File->getName().str().c_str(), nullptr)};
    if (realIncludePath.find("/usr/") != std::string::npos)
      return;
    // 获取#include指令的结束位置
    SourceLocation EndLoc = Lexer::getLocForEndOfToken(FilenameRange.getEnd(),
                                                       0, SM, LangOptions());
    // 获取完整指令的源码范围
    CharSourceRange FullRange = CharSourceRange::getCharRange(HashLoc, EndLoc);
    // 提取源码文本
    StringRef Text = Lexer::getSourceText(FullRange, SM, LangOptions());
    ss << "{\"type\":\"includedefine\",\"file\":\"" << PLoc.getFilename()
       << "\",\"line\":" << PLoc.getLine() << ",\"column\":" << PLoc.getColumn()
       << ",\"includefilepath\":\"" << realIncludePath << "\",\"stmt\":\""
       << Text.str() << "\"}";
    printf("%s\n", ss.str().c_str());
  }
};

class AnalysisAction : public PreprocessorFrontendAction {
protected:
  void ExecuteAction() override {
    Preprocessor &PP = getCompilerInstance().getPreprocessor();
    PP.addPPCallbacks(std::make_unique<AnalysisCallback>(PP));
    // Process all tokens to trigger preprocessing callbacks
    PP.EnterMainSourceFile();
    Token Tok;
    do {
      PP.Lex(Tok);
    } while (Tok.isNot(tok::eof));
  }
};

// AST访问器类
class AnalysisVisitor : public RecursiveASTVisitor<AnalysisVisitor> {
public:
  explicit AnalysisVisitor(ASTContext &Context, SourceManager &SM)
      : Context(Context), SM(SM) {}

  // 捕获命名空间定义
  bool VisitNamespaceDecl(NamespaceDecl *D) {
    // report("NamespaceDef", D->getNameAsString(), D->getSourceRange());
    SourceLocation Loc = SM.getSpellingLoc(D->getLocation());
    if (Loc.isInvalid() || SM.isInSystemHeader(Loc) ||
        !SM.isWrittenInMainFile(Loc))
      return true;
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    std::stringstream ss;
    ss << "{\"type\":\"namespacedeclare\",\"file\":\"" << PLoc.getFilename()
       << "\",\"line\":" << PLoc.getLine() << ",\"column\":" << PLoc.getColumn()
       << ",\"namespace\":\"" << D->getName().str() << "\"}";
    printf("%s\n", ss.str().c_str());
    return true;
  }

  // 捕获using namespace声明
  bool VisitUsingDirectiveDecl(UsingDirectiveDecl *D) {
    if (const NamespaceDecl *ND = D->getNominatedNamespace()) {
      // report("UsingNamespace", ND->getNameAsString(), D->getSourceRange());
      SourceLocation Loc = SM.getSpellingLoc(D->getLocation());
      if (Loc.isInvalid() || SM.isInSystemHeader(Loc) ||
          !SM.isWrittenInMainFile(Loc))
        return true;
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      std::stringstream ss;
      ss << "{\"type\":\"usingnamespace\",\"file\":\"" << PLoc.getFilename()
         << "\",\"line\":" << PLoc.getLine()
         << ",\"column\":" << PLoc.getColumn() << "}";
      printf("%s\n", ss.str().c_str());
    }
    return true;
  }

  // 捕获using声明
  bool VisitUsingDecl(UsingDecl *D) {
    // std::string name;
    // llvm::raw_string_ostream os(name);
    // D->getQualifier()->print(os, D->getASTContext().getPrintingPolicy());
    // os << D->getNameAsString();
    // report("UsingDecl", os.str(), D->getSourceRange());
    SourceLocation Loc = SM.getSpellingLoc(D->getLocation());
    if (Loc.isInvalid() || SM.isInSystemHeader(Loc) ||
        !SM.isWrittenInMainFile(Loc))
      return true;
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    std::stringstream ss;
    ss << "{\"type\":\"using\",\"file\":\"" << PLoc.getFilename()
       << "\",\"line\":" << PLoc.getLine() << ",\"column\":" << PLoc.getColumn()
       << "}";
    printf("%s\n", ss.str().c_str());
    return true;
  }

  // 捕获函数定义
  bool VisitFunctionDecl(FunctionDecl *FD) {
    // if (!FD->isThisDeclarationADefinition()) return true;     // 跳过声明
    // if (SM.isInSystemHeader(FD->getLocation())) return true;  // 过滤系统函数
    // analyzeFunction(FD);
    SourceLocation Loc = SM.getSpellingLoc(FD->getLocation());
    if (Loc.isInvalid() || SM.isInSystemHeader(Loc) ||
        !SM.isWrittenInMainFile(Loc))
      return true;
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    const CompoundStmt *Body = dyn_cast_or_null<CompoundStmt>(FD->getBody());
    std::stringstream ss;
    if (Body != nullptr) {
      PresumedLoc PLocImp = SM.getPresumedLoc(Body->getLBracLoc());
      ss << "{\"type\":\"functionimplement\",\"file\":\""
         << PLocImp.getFilename() << "\",\"line\":" << PLocImp.getLine()
         << ",\"startcolumn\":" << PLoc.getColumn()
         << ",\"endcolumn\":" << PLocImp.getColumn() - 1
         << ",\"functionname\":\"" << FD->getNameAsString() << "\"}";
    } else {
      ss << "{\"type\":\"functiondefine\",\"file\":\"" << PLoc.getFilename()
         << "\",\"line\":" << PLoc.getLine()
         << ",\"column\":" << PLoc.getColumn() << ",\"functionname\":\""
         << FD->getNameAsString() << "\"}";
    }
    printf("%s\n", ss.str().c_str());
    return true;
  }

  // 捕获普通函数调用
  bool VisitCallExpr(CallExpr *CE) {
    if (FunctionDecl *FD = CE->getDirectCallee()) {
      // reportCall(FD, CE->getSourceRange());
      SourceLocation Loc = SM.getSpellingLoc(CE->getSourceRange().getBegin());
      SourceLocation FDLoc = SM.getSpellingLoc(FD->getLocation());
      if (FDLoc.isInvalid() || SM.isInSystemHeader(FDLoc))
        return true;
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      PresumedLoc PFDLoc = SM.getPresumedLoc(FDLoc);
      SourceLocation EndLoc = Lexer::getLocForEndOfToken(
          CE->getSourceRange().getEnd(), 0, SM, LangOptions());
      CharSourceRange FullRange = CharSourceRange::getCharRange(
          CE->getSourceRange().getBegin(), EndLoc);
      StringRef Text = Lexer::getSourceText(FullRange, SM, LangOptions());
      std::stringstream ss;
      ss << "{\"type\":\"functioncall\",\"file\":\"" << PLoc.getFilename()
         << "\",\"line\":" << PLoc.getLine()
         << ",\"column\":" << PLoc.getColumn() << ",\"functiondefinefile\":\""
         << PFDLoc.getFilename() << "\",\"stmt\":\"" << Text.str() << "\"}";
      printf("%s\n", ss.str().c_str());
    }
    return true;
  }

  // 捕获成员函数调用
  bool VisitCXXMemberCallExpr(CXXMemberCallExpr *MCE) {
    if (CXXMethodDecl *MD = MCE->getMethodDecl()) {
      // reportCall(MD, MCE->getSourceRange());
      SourceLocation Loc = SM.getSpellingLoc(MCE->getSourceRange().getBegin());
      SourceLocation FDLoc = SM.getSpellingLoc(MD->getLocation());
      if (FDLoc.isInvalid() || SM.isInSystemHeader(FDLoc))
        return true;
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      PresumedLoc PFDLoc = SM.getPresumedLoc(FDLoc);
      std::stringstream ss;
      ss << "{\"type\":\"memberfunctioncall\",\"file\":\"" << PLoc.getFilename()
         << "\",\"line\":" << PLoc.getLine()
         << ",\"column\":" << PLoc.getColumn() << ",\"functiondefinefile\":\""
         << PFDLoc.getFilename() << "\"}";
      printf("%s\n", ss.str().c_str());
    }
    return true;
  }

  // 捕获类/结构体/联合体声明
  bool VisitCXXRecordDecl(CXXRecordDecl *D) {
    SourceLocation Loc = SM.getSpellingLoc(D->getLocation());
    if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
      return true;
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    std::stringstream ss;
    ss << "{\"type\":\"classdefine\",\"file\":\"" << PLoc.getFilename()
       << "\",\"line\":" << PLoc.getLine() << ",\"column\":" << PLoc.getColumn()
       << ",\"definename\":\"" << D->getNameAsString() << "\"}";
    printf("%s\n", ss.str().c_str());
    return true;
  }

  // 捕获全局变量、局部变量、静态变量
  bool VisitVarDecl(VarDecl *VD) {
    SourceLocation Loc = SM.getSpellingLoc(VD->getLocation());
    if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
      return true;
    QualType Type = VD->getType();
    std::string typeName = Type.getAsString();
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    std::stringstream ss;
    ss << "{\"type\":\"variabledefine\",\"file\":\"" << PLoc.getFilename()
       << "\",\"line\":" << PLoc.getLine() << ",\"column\":" << PLoc.getColumn()
       << ",\"definetype\":\"" << typeName << "\"}";
    printf("%s\n", ss.str().c_str());
    return true;
  }

  // 捕获类成员变量
  bool VisitFieldDecl(FieldDecl *FD) {
    SourceLocation Loc = SM.getSpellingLoc(FD->getLocation());
    if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
      return true;
    QualType Type = FD->getType();
    std::string typeName = Type.getAsString();
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    std::stringstream ss;
    ss << "{\"type\":\"fielddefine\",\"file\":\"" << PLoc.getFilename()
       << "\",\"line\":" << PLoc.getLine() << ",\"column\":" << PLoc.getColumn()
       << ",\"definetype\":\"" << typeName << "\"}";
    printf("%s\n", ss.str().c_str());
    return true;
  }

  // 捕获函数参数
  bool VisitParmVarDecl(ParmVarDecl *PD) {
    SourceLocation Loc = SM.getSpellingLoc(PD->getLocation());
    if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
      return true;
    QualType Type = PD->getType();
    std::string typeName = Type.getAsString();
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    std::stringstream ss;
    ss << "{\"type\":\"parameterdefine\",\"file\":\"" << PLoc.getFilename()
       << "\",\"line\":" << PLoc.getLine() << ",\"column\":" << PLoc.getColumn()
       << ",\"definetype\":\"" << typeName << "\"}";
    printf("%s\n", ss.str().c_str());
    return true;
  }

  bool VisitDeclRefExpr(DeclRefExpr *DRE) {
    if (auto *NNS = DRE->getQualifier()) {
      analyzeNestedNameSpecifier(NNS, DRE->getLocation());
    }
    return true;
  }

  bool VisitTypeLoc(TypeLoc TL) {
    if (auto elaboratedLoc = TL.getAs<ElaboratedTypeLoc>()) {
      NestedNameSpecifierLoc qualifierLoc = elaboratedLoc.getQualifierLoc();
      if (auto *NNS = qualifierLoc.getNestedNameSpecifier()) {
        analyzeNestedNameSpecifier(NNS, qualifierLoc.getBeginLoc());
      }
    }
    return true;
  }

  void analyzeNestedNameSpecifier(NestedNameSpecifier *NNS,
                                  SourceLocation UseLoc) {
    auto getLocationString = [&](SourceLocation Loc) -> std::string {
      if (Loc.isInvalid())
        return "<unknown>";
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      return std::string(PLoc.getFilename()) + ":" +
             std::to_string(PLoc.getLine()) + ":" +
             std::to_string(PLoc.getColumn());
    };
    auto getScopeTypeName = [](const Decl *D) -> std::string {
      if (isa<NamespaceDecl>(D))
        return "Namespace";
      if (isa<CXXRecordDecl>(D))
        return "Class/Struct";
      if (isa<EnumDecl>(D))
        return "Enum";
      return "Unknown";
    };

    if (!NNS)
      return;
    // 遍历嵌套作用域
    while (NNS) {
      if (NNS->getKind() == NestedNameSpecifier::TypeSpec ||
          NNS->getKind() == NestedNameSpecifier::Namespace) {
        // 获取作用域声明
        const Decl *ScopeDecl = nullptr;
        if (const auto *T = NNS->getAsType()) {
          if (const auto *TD = T->getAsTagDecl()) {
            ScopeDecl = TD;
          }
        } else if (NNS->getKind() == NestedNameSpecifier::Namespace) {
          ScopeDecl = NNS->getAsNamespace();
        }
        // 输出作用域信息
        if (ScopeDecl && !SM.isInSystemHeader(ScopeDecl->getLocation())) {
          std::string ScopeType = getScopeTypeName(ScopeDecl);
          std::string DeclLocation =
              getLocationString(ScopeDecl->getLocation());
          std::string UseLocation = getLocationString(UseLoc);
          llvm::outs() << "Found Scope Resolution Operator (::)\n"
                       << "  Scope Type:    " << ScopeType << "\n"
                       << "  Declared At:   " << DeclLocation << "\n"
                       << "  Used At:       " << UseLocation << "\n\n";
        }
      }
      NNS = NNS->getPrefix();
    }
  }

private:
  ASTContext &Context;
  SourceManager &SM;
};

// AST消费者类
class AnalysisConsumer : public ASTConsumer {
public:
  explicit AnalysisConsumer(ASTContext &Context, SourceManager &SM)
      : Visitor(Context, SM) {}
  void HandleTranslationUnit(ASTContext &Context) override {
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }

private:
  AnalysisVisitor Visitor;
};

// FrontendAction类
class NamespaceAction : public ASTFrontendAction {
public:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef File) override {
    return std::make_unique<AnalysisConsumer>(CI.getASTContext(),
                                              CI.getSourceManager());
  }
};

// 定义命令行选项
static llvm::cl::OptionCategory Category("Analysis Options");
int main(int argc, const char **argv) {
  // 解析命令行参数和编译命令
  auto ExpectedParser = CommonOptionsParser::create(argc, argv, Category);
  if (!ExpectedParser) {
    llvm::errs() << llvm::toString(ExpectedParser.takeError());
    return 1;
  }
  // 创建工具并运行
  ClangTool Tool(ExpectedParser->getCompilations(),
                 ExpectedParser->getSourcePathList());
  Tool.run(newFrontendActionFactory<AnalysisAction>().get());
  Tool.run(newFrontendActionFactory<NamespaceAction>().get());
  return 0;
}




#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/MacroArgs.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/FrontendAction.h>

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"

#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

using namespace clang;
using namespace clang::tooling;
using namespace clang::ast_matchers;
using namespace llvm;

static std::string getPath(const std::string &path) {
  auto realPathPtr = realpath(path.c_str(), nullptr);
  if (realPathPtr == nullptr)
    return "";
  std::string realPath{realPathPtr};
  free(realPathPtr);
  return realPath;
}

static bool skipPath(const std::string &realPath) {
  if (realPath.empty())
    return true;
  if (realPath.find("/usr/") != std::string::npos) {
    return true;
  }
  if (realPath.find("/ThirdPartyLib/") != std::string::npos) {
    return true;
  }
  if (realPath.find("/open/") != std::string::npos) {
    return true;
  }
  if (realPath.find("<") == 0) {
    return true;
  }
  return false;
}

static std::string escapeJsonString(const std::string &input) {
  std::ostringstream ss;
  for (char c : input) {
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

static std::string option;

class AnalysisPPCallback : public PPCallbacks {
  Preprocessor &PP;

public:
  AnalysisPPCallback(Preprocessor &PP) : PP(PP) {}

  // void MacroDefined(const Token &MacroNameTok,
  //                   const MacroDirective *MD) override {
  //   const MacroInfo *MI = MD->getMacroInfo();
  //   SourceManager &SM = PP.getSourceManager();
  //   SourceLocation Loc = SM.getSpellingLoc(MacroNameTok.getLocation());
  //   if (Loc.isInvalid() || SM.isInSystemHeader(Loc)) return;
  //   PresumedLoc PLoc = SM.getPresumedLoc(Loc);
  //   if (skipPath(PLoc.getFilename())) return;
  //   std::string filename{PLoc.getFilename() != nullptr ? PLoc.getFilename() :
  //   ""}; if (filename.empty() || filename.find("<") != std::string::npos)
  //   return; SourceLocation BeginLoc =
  //   Lexer::GetBeginningOfToken(MI->getDefinitionLoc(), SM, LangOptions());
  //   SourceLocation EndLoc =
  //   Lexer::getLocForEndOfToken(MI->getDefinitionEndLoc(), 0, SM,
  //   LangOptions()); CharSourceRange FullRange =
  //   CharSourceRange::getCharRange(BeginLoc, EndLoc); StringRef Text =
  //   Lexer::getSourceText(FullRange, SM, LangOptions()); std::stringstream ss;
  //   ss << "{\"type\":\"macrodefine\",\"file\":\"" << PLoc.getFilename() <<
  //   "\",\"line\":" << PLoc.getLine() << ",\"column\":" << PLoc.getColumn() <<
  //   ",\"macroname\":\"" << MacroNameTok.getIdentifierInfo()->getName().str()
  //   << "\",\"stmt\":\"" << escapeJsonString(Text.str()) << "\"}";
  //   printf("%s\n", ss.str().c_str());
  // }

  void MacroExpands(const Token &MacroNameTok,
                    const MacroDefinition &MacroDefinition, SourceRange Range,
                    const MacroArgs *Args) override {
    if (option.find("MacroExpands|") == std::string::npos)
      return;
    SourceManager &SM = PP.getSourceManager();
    SourceLocation Loc = SM.getSpellingLoc(MacroNameTok.getLocation());
    if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
      return;
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    auto filePath = getPath(PLoc.getFilename());
    if (skipPath(filePath))
      return;
    // SourceLocation BeginLoc = Lexer::GetBeginningOfToken(
    //     MacroNameTok.getLocation(), SM, LangOptions());
    // SourceLocation EndLoc =
    //     Lexer::getLocForEndOfToken(Range.getEnd(), 0, SM, LangOptions());
    // CharSourceRange FullRange = CharSourceRange::getCharRange(BeginLoc,
    // EndLoc); StringRef Text = Lexer::getSourceText(FullRange, SM,
    // LangOptions());
    std::stringstream ss;
    ss << "{\"type\":\"MacroExpands\",\"file\":\"" << filePath
       << "\",\"line\":" << PLoc.getLine() << ",\"column\":" << PLoc.getColumn()
       << ",\"macroname\":\""
       << MacroNameTok.getIdentifierInfo()->getName().str()
       << "\",\"stmt\":\""
       /*<< escapeJsonString(Text.str())*/
       << "\"}";
    printf("%s\n", ss.str().c_str());
  }

  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange,
                          OptionalFileEntryRef File, StringRef SearchPath,
                          StringRef RelativePath, const Module *Imported,
                          SrcMgr::CharacteristicKind FileType) override {
    if (option.find("InclusionDirective|") == std::string::npos)
      return;
    SourceManager &SM = PP.getSourceManager();
    SourceLocation Loc = SM.getSpellingLoc(IncludeTok.getLocation());
    if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
      return;
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    auto filePath = getPath(PLoc.getFilename());
    if (skipPath(filePath))
      return;
    std::string includeFilePath{getPath(File->getName().str())};
    if (skipPath(includeFilePath))
      return;
    // 获取#include指令的结束位置
    SourceLocation EndLoc = Lexer::getLocForEndOfToken(FilenameRange.getEnd(),
                                                       0, SM, LangOptions());
    // 获取完整指令的源码范围
    CharSourceRange FullRange = CharSourceRange::getCharRange(HashLoc, EndLoc);
    // 提取源码文本
    StringRef Text = Lexer::getSourceText(FullRange, SM, LangOptions());
    std::stringstream ss;
    ss << "{\"type\":\"InclusionDirective\",\"file\":\"" << filePath
       << "\",\"line\":" << PLoc.getLine() << ",\"column\":" << PLoc.getColumn()
       << ",\"includefilepath\":\"" << includeFilePath << "\",\"stmt\":\""
       << escapeJsonString(Text.str()) << "\"}";
    printf("%s\n", ss.str().c_str());
  }
};

class PPAnalysisAction : public PreprocessorFrontendAction {
protected:
  void ExecuteAction() override {
    Preprocessor &PP = getCompilerInstance().getPreprocessor();
    PP.addPPCallbacks(std::make_unique<AnalysisPPCallback>(PP));
    // Process all tokens to trigger preprocessing callbacks
    PP.EnterMainSourceFile();
    Token Tok;
    do {
      PP.Lex(Tok);
    } while (Tok.isNot(tok::eof));
  }
};

static void process(const MatchFinder::MatchResult &Result, SourceManager &SM) {
  if (const auto *M = Result.Nodes.getNodeAs<NamespaceDecl>("check")) {
    std::string file, Dfile;
    int line = 0, column = 0;
    int Dline = 0, Dcolumn = 0;
    SourceLocation Loc = SM.getSpellingLoc(M->getLocation());
    if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
      return;
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    if (PLoc.isValid()) {
      file = getPath(PLoc.getFilename());
      line = PLoc.getLine();
      column = PLoc.getColumn();
    }
    std::cout << "NamespaceDecl " << file << " " << line << " " << column << "\n";
  } else if (const auto *M = Result.Nodes.getNodeAs<TypeLoc>("check")) {
    std::string file, Dfile;
    int line = 0, column = 0;
    int Dline = 0, Dcolumn = 0;
    SourceLocation Loc = SM.getSpellingLoc(M->getBeginLoc());
    if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
      return;
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    if (PLoc.isValid()) {
      file = getPath(PLoc.getFilename());
      line = PLoc.getLine();
      column = PLoc.getColumn();
    }
    std::cout << "TypeLoc " << file << " " << line << " " << column << "\n";
  } else if (const auto *M = Result.Nodes.getNodeAs<DeclRefExpr>("check")) {
    std::string file, Dfile;
    int line = 0, column = 0;
    int Dline = 0, Dcolumn = 0;
    SourceLocation Loc = SM.getSpellingLoc(M->getBeginLoc());
    if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
      return;
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    if (PLoc.isValid()) {
      file = getPath(PLoc.getFilename());
      line = PLoc.getLine();
      column = PLoc.getColumn();
    }
    std::cout << "DeclRefExpr " << file << " " << line << " " << column << "\n";
  } else if (const auto *M = Result.Nodes.getNodeAs<UsingDirectiveDecl>("check")) {
    std::string file, Dfile;
    int line = 0, column = 0;
    int Dline = 0, Dcolumn = 0;
    SourceLocation Loc = SM.getSpellingLoc(M->getBeginLoc());
    if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
      return;
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    if (PLoc.isValid()) {
      file = getPath(PLoc.getFilename());
      line = PLoc.getLine();
      column = PLoc.getColumn();
    }
    std::cout << "UsingDirectiveDecl " << file << " " << line << " " << column << "\n";
  } else if (const auto *M = Result.Nodes.getNodeAs<QualType>("check")) {
    std::string file, Dfile;
    int line = 0, column = 0;
    int Dline = 0, Dcolumn = 0;
    // SourceLocation Loc = SM.getSpellingLoc(M->getBeginLoc());
    // if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
    //   return;
    // PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    // if (PLoc.isValid()) {
    //   file = getPath(PLoc.getFilename());
    //   line = PLoc.getLine();
    //   column = PLoc.getColumn();
    // }
    // std::cout << "QualType " << file << " " << line << " " << column << "\n";
  } else if (const auto *M = Result.Nodes.getNodeAs<NamedDecl>("check")) {
    std::string file, Dfile;
    int line = 0, column = 0;
    int Dline = 0, Dcolumn = 0;
    SourceLocation Loc = SM.getSpellingLoc(M->getBeginLoc());
    if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
      return;
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    if (PLoc.isValid()) {
      file = getPath(PLoc.getFilename());
      line = PLoc.getLine();
      column = PLoc.getColumn();
    }
    std::cout << "TypeLoc " << file << " " << line << " " << column << "\n";
  } else if (const auto *M = Result.Nodes.getNodeAs<Stmt>("check")) {
    std::string file, Dfile;
    int line = 0, column = 0;
    int Dline = 0, Dcolumn = 0;
    SourceLocation Loc = SM.getSpellingLoc(M->getBeginLoc());
    if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
      return;
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    if (PLoc.isValid()) {
      file = getPath(PLoc.getFilename());
      line = PLoc.getLine();
      column = PLoc.getColumn();
    }
    std::cout << "Stmt " << file << " " << line << " " << column << "\n";
  }
}

class AnalysisMatchCallback : public MatchFinder::MatchCallback {
public:
  explicit AnalysisMatchCallback(ASTContext &Context, SourceManager &SM)
      : Context(Context), SM(SM) {}

  virtual void run(const MatchFinder::MatchResult &Result) {
    if (const auto *ND =
            Result.Nodes.getNodeAs<NamespaceDecl>("NamespaceDecl|")) {
      std::string file, Dfile;
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
      std::stringstream ss;
      ss << "{\"type\":\"NamespaceDecl\",\"file\":\"" << file
         << "\",\"line\":" << line << ",\"column\":" << column
         << ",\"namespace\":\"" << ND->getName().str() << "\"}";
      printf("%s\n", ss.str().c_str());
    } else if (const auto *TL = Result.Nodes.getNodeAs<TypeLoc>("TypeLoc|")) {
      std::string file, Dfile, stmt, Dstmt;
      int line = 0, column = 0;
      int Dline = 0, Dcolumn = 0;
      SourceLocation Loc = SM.getSpellingLoc(TL->getBeginLoc());
      if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
        return;
      {
        auto FullRange =
            CharSourceRange::getCharRange(TL->getBeginLoc(), TL->getEndLoc());
        auto Text = Lexer::getSourceText(FullRange, SM, LangOptions());
        stmt = escapeJsonString(Text.str());
      }
      SourceLocation DLoc = Loc;
      if (auto ETL = TL->getAs<ElaboratedTypeLoc>()) {
        auto Q = ETL.getQualifierLoc();
        DLoc = SM.getSpellingLoc(Q.getBeginLoc());
        auto FullRange =
            CharSourceRange::getCharRange(Q.getBeginLoc(), Q.getEndLoc());
        auto Text = Lexer::getSourceText(FullRange, SM, LangOptions());
        Dstmt = escapeJsonString(Text.str());
      } else if (auto RTL = TL->getAs<RecordTypeLoc>()) {
        auto *D = RTL.getDecl();
        DLoc = SM.getSpellingLoc(D->getBeginLoc());
        auto FullRange =
            CharSourceRange::getCharRange(D->getBeginLoc(), D->getEndLoc());
        auto Text = Lexer::getSourceText(FullRange, SM, LangOptions());
        Dstmt = escapeJsonString(Text.str());
      }
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      if (PLoc.isValid()) {
        file = getPath(PLoc.getFilename());
        line = PLoc.getLine();
        column = PLoc.getColumn();
      }
      if (!file.empty() && skipPath(file))
        return;
      PresumedLoc DPLoc = SM.getPresumedLoc(DLoc);
      if (DPLoc.isValid()) {
        Dfile = getPath(DPLoc.getFilename());
        Dline = DPLoc.getLine();
        Dcolumn = DPLoc.getColumn();
      }
      if (!Dfile.empty() && skipPath(Dfile))
        return;
      std::stringstream ss;
      ss << "{\"type\":\"TypeLoc\",\"file\":\"" << file
         << "\",\"line\":" << line << ",\"column\":" << column << ",\"stmt\":\""
         << stmt << "\",\"definename\":\"" << TL->getType()->getTypeClassName()
         << "\",\"dfile\":\"" << Dfile << "\",\"dline\":" << Dline
         << ",\"dcolumn\":" << Dcolumn << ",\"dstmt\":\"" << Dstmt << "\"}";
      printf("%s\n", ss.str().c_str());
    } else if (const auto *DRE =
                   Result.Nodes.getNodeAs<DeclRefExpr>("DeclRefExpr|")) {
      std::string file, Dfile, stmt, Dstmt;
      int line = 0, column = 0;
      int Dline = 0, Dcolumn = 0;
      SourceLocation Loc = SM.getSpellingLoc(DRE->getBeginLoc());
      if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
        return;
      {
        auto FullRange =
            CharSourceRange::getCharRange(DRE->getBeginLoc(), DRE->getEndLoc());
        auto Text = Lexer::getSourceText(FullRange, SM, LangOptions());
        stmt = escapeJsonString(Text.str());
      }
      const auto *D = DRE->getDecl();
      auto DLoc = SM.getSpellingLoc(D->getBeginLoc());
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      if (PLoc.isValid()) {
        file = getPath(PLoc.getFilename());
        line = PLoc.getLine();
        column = PLoc.getColumn();
      }
      if (!file.empty() && skipPath(file))
        return;
      PresumedLoc DPLoc = SM.getPresumedLoc(DLoc);
      if (DPLoc.isValid()) {
        Dfile = getPath(DPLoc.getFilename());
        Dline = DPLoc.getLine();
        Dcolumn = DPLoc.getColumn();
      }
      if (!Dfile.empty() && skipPath(Dfile))
        return;
      {
        auto FullRange =
            CharSourceRange::getCharRange(D->getBeginLoc(), D->getEndLoc());
        auto Text = Lexer::getSourceText(FullRange, SM, LangOptions());
        Dstmt = escapeJsonString(Text.str());
      }
      std::stringstream ss;
      ss << "{\"type\":\"DeclRefExpr\",\"file\":\"" << file
         << "\",\"line\":" << line << ",\"column\":" << column << ",\"stmt\":\""
         << stmt << "\",\"definename\":\"" << DRE->getType()->getTypeClassName()
         << "\",\"dfile\":\"" << Dfile << "\",\"dline\":" << Dline
         << ",\"dcolumn\":" << Dcolumn << ",\"dstmt\":\"" << Dstmt << "\"}";
      printf("%s\n", ss.str().c_str());
    } else if (const auto *FD =
                   Result.Nodes.getNodeAs<FunctionDecl>("FunctionDecl|")) {
      std::string file, Dfile, stmt, Dstmt;
      int line = 0, column = 0;
      int Dline = 0, Dcolumn = 0;
      SourceLocation Loc = SM.getSpellingLoc(FD->getBeginLoc());
      if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
        return;
      const CompoundStmt *Body = dyn_cast_or_null<CompoundStmt>(FD->getBody());
      if (Body != nullptr) {
        auto FullRange = CharSourceRange::getCharRange(
            FD->getBeginLoc(), FD->getBody()->getBeginLoc());
        auto Text = Lexer::getSourceText(FullRange, SM, LangOptions());
        stmt = escapeJsonString(Text.str());
      } else {
        auto FullRange = CharSourceRange::getCharRange(
            FD->getBeginLoc(), FD->getEndLoc());
        auto Text = Lexer::getSourceText(FullRange, SM, LangOptions());
        stmt = escapeJsonString(Text.str());
      }
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      if (PLoc.isValid()) {
        file = getPath(PLoc.getFilename());
        line = PLoc.getLine();
        column = PLoc.getColumn();
      }
      if (!file.empty() && skipPath(file))
        return;
      // const DeclContext *DC = FD->getDeclContext();
      // while (DC) {
      //   // printf("getDeclKindName %s\n", DC->getDeclKindName());
      //   if (const auto *NS = dyn_cast<NamespaceDecl>(DC)) {
      //     std::string QualifierName = NS->getNameAsString();
      //     Dstmt = QualifierName + "::" + Dstmt;
      //   }
      //   DC = DC->getParent();
      // }
      SourceLocation BeginLoc = Lexer::GetBeginningOfToken(
          FD->getQualifierLoc().getBeginLoc(), SM, LangOptions());
      SourceLocation EndLoc = Lexer::getLocForEndOfToken(
          FD->getQualifierLoc().getEndLoc(), 0, SM, LangOptions());
      CharSourceRange FullRange =
          CharSourceRange::getCharRange(BeginLoc, EndLoc);
      StringRef Text = Lexer::getSourceText(FullRange, SM, LangOptions());
      Dstmt = Text.str();
      PresumedLoc DPLoc =
          SM.getPresumedLoc(FD->getQualifierLoc().getBeginLoc());
      if (DPLoc.isValid()) {
        Dfile = getPath(DPLoc.getFilename());
        Dline = DPLoc.getLine();
        Dcolumn = DPLoc.getColumn();
      }
      // if (NestedNameSpecifier *NNS = FD->getQualifier()) {
      //   while (NNS) {
      //     // printf("getKind %d\n", NNS->getKind());
      //     if (NNS->getKind() == NestedNameSpecifier::Namespace) {
      //       auto NS = NNS->getAsNamespace();
      //       std::string QualifierName = NS->getNameAsString();
      //       Dstmt = QualifierName + "::" + Dstmt;
      //     } else if (NNS->getKind() == NestedNameSpecifier::NamespaceAlias) {
      //       auto NS = NNS->getAsNamespaceAlias();
      //       std::string QualifierName = NS->getNameAsString();
      //       Dstmt = QualifierName + "::" + Dstmt;
      //     } else if (NNS->getKind() == NestedNameSpecifier::TypeSpec) {
      //       auto T = NNS->getAsType();
      //       if (std::string{T->getTypeClassName()} == "Record") {
      //         std::string QualifierName =
      //         T->getAsRecordDecl()->getNameAsString(); Dstmt = QualifierName
      //         + "::" + Dstmt;
      //       }
      //     }
      //     NNS = NNS->getPrefix(); // 处理嵌套命名空间（如A::B::func）
      //   }
      // }
      std::stringstream ss;
      ss << "{\"type\":\"FunctionDecl\",\"file\":\"" << file
         << "\",\"line\":" << line << ",\"column\":" << column << ",\"stmt\":\""
         << stmt << "\",\"definename\":\""
         << (!FD->isThisDeclarationADefinition() || FD->getBody() == nullptr ? "functiondecl" : "functionimp")
         << "\",\"dfile\":\"" << Dfile << "\",\"dline\":" << Dline
         << ",\"dcolumn\":" << Dcolumn << ",\"dstmt\":\"" << Dstmt << "\"}";
      printf("%s\n", ss.str().c_str());
    } else if (const auto *S = Result.Nodes.getNodeAs<Stmt>("Stmt|")) {
      std::string file, Dfile, stmt, Dstmt;
      int line = 0, column = 0;
      int Dline = 0, Dcolumn = 0;
      SourceLocation Loc = SM.getSpellingLoc(S->getBeginLoc());
      if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
        return;
      {
        auto FullRange =
            CharSourceRange::getCharRange(S->getBeginLoc(), S->getEndLoc());
        auto Text = Lexer::getSourceText(FullRange, SM, LangOptions());
        stmt = escapeJsonString(Text.str());
      }
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      if (PLoc.isValid()) {
        file = getPath(PLoc.getFilename());
        line = PLoc.getLine();
        column = PLoc.getColumn();
      }
      if (!file.empty() && skipPath(file))
        return;
      std::stringstream ss;
      ss << "{\"type\":\"Stmt\",\"file\":\"" << file << "\",\"line\":" << line
         << ",\"column\":" << column << ",\"stmt\":\"" << stmt
         << "\",\"definename\":\"" << S->getStmtClassName() << "\",\"dfile\":\""
         << Dfile << "\",\"dline\":" << Dline << ",\"dcolumn\":" << Dcolumn
         << ",\"dstmt\":\"" << Dstmt << "\"}";
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
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef file) override {
    MatchFinder Finder;
    auto *Callback{
        new AnalysisMatchCallback{CI.getASTContext(), CI.getSourceManager()}};

    // Finder.addMatcher(clang::ast_matchers::namespaceDecl().bind("check"),
    //                   Callback);
    // Finder.addMatcher(clang::ast_matchers::nestedNameSpecifierLoc().bind(
    //                       "nestedNameSpecifierLoc"),
    //                   Callback);
    // Finder.addMatcher(clang::ast_matchers::declRefExpr().bind("check"),
    //                   Callback);
    // Finder.addMatcher(clang::ast_matchers::usingDirectiveDecl().bind("check"),
    //                   Callback);
    // Finder.addMatcher(clang::ast_matchers::qualType().bind("check"), Callback);
    // Finder.addMatcher(clang::ast_matchers::namedDecl().bind("check"), Callback);
    // Finder.addMatcher(clang::ast_matchers::typeLoc().bind("check"), Callback);
    // Finder.addMatcher(clang::ast_matchers::stmt().bind("check"), Callback);

    if (option.find("NamespaceDecl|") != std::string::npos) {
      Finder.addMatcher(
          clang::ast_matchers::namespaceDecl().bind("NamespaceDecl|"),
          Callback);
    } else if (option.find("TypeLoc|") != std::string::npos) {
      Finder.addMatcher(clang::ast_matchers::typeLoc().bind("TypeLoc|"),
                        Callback);
    } else if (option.find("DeclRefExpr|") != std::string::npos) {
      Finder.addMatcher(clang::ast_matchers::declRefExpr().bind("DeclRefExpr|"),
                        Callback);
    } else if (option.find("FunctionDecl|") != std::string::npos) {
      Finder.addMatcher(
          clang::ast_matchers::functionDecl().bind("FunctionDecl|"), Callback);
    } else if (option.find("Stmt|") != std::string::npos) {
      Finder.addMatcher(clang::ast_matchers::stmt().bind("Stmt|"), Callback);
    }
    return std::make_unique<AnalysisConsumer>(Finder);
  }
};

// 定义命令行选项
static llvm::cl::OptionCategory Category("Analysis Options");
int main(int argc, const char **argv) {
  if (getenv("ANALYZE_CMD") != nullptr) {
    option = getenv("ANALYZE_CMD");
  }
  // 解析命令行参数和编译命令
  auto ExpectedParser = CommonOptionsParser::create(argc, argv, Category);
  if (!ExpectedParser) {
    llvm::errs() << llvm::toString(ExpectedParser.takeError());
    return 1;
  }
  // 创建工具并运行
  ClangTool Tool(ExpectedParser->getCompilations(),
                 ExpectedParser->getSourcePathList());
  Tool.run(newFrontendActionFactory<PPAnalysisAction>().get());
  Tool.run(newFrontendActionFactory<AnalysisAction>().get());
  return 0;
}

// // AST访问器类
// class AnalysisVisitor : public RecursiveASTVisitor<AnalysisVisitor> {
// public:
//   explicit AnalysisVisitor(ASTContext &Context, SourceManager &SM)
//       : Context(Context), SM(SM) {}

//   bool VisitStmt(Stmt *stmt) {
//     if (option.find("STATEMENT|") == std::string::npos)
//       return true;
//     SourceLocation loc = stmt->getBeginLoc();
//     if (loc.isInvalid() || SM.isInSystemHeader(loc))
//       return true;
//     PresumedLoc PLoc = SM.getPresumedLoc(loc);
//     auto filePath = getPath(PLoc.getFilename());
//     if (skipPath(filePath))
//       return true;
//     PresumedLoc declarePLoc;
//     std::string declareFilePath;
//     SourceRange range = stmt->getSourceRange();
//     StringRef Text =
//     Lexer::getSourceText(CharSourceRange::getTokenRange(range),
//                                           SM, Context.getLangOpts());
//     std::stringstream ss;
//     switch (stmt->getStmtClass()) {
//     case Stmt::DeclRefExprClass:
//       if (const DeclRefExpr *expr = dyn_cast<DeclRefExpr>(stmt)) {
//         const Decl *D = expr->getDecl();
//         if (D) {
//           auto declareLoc = D->getLocation();
//           if (declareLoc.isValid()) {
//             declarePLoc = SM.getPresumedLoc(declareLoc);
//             if (declarePLoc.isValid()) {
//               declareFilePath = getPath(declarePLoc.getFilename());
//               if (skipPath(declareFilePath))
//                 return true;
//             }
//           }
//         }
//         ss << "{\"type\":\"typereference\",\"file\":\"" << filePath
//            << "\",\"line\":" << PLoc.getLine()
//            << ",\"column\":" << PLoc.getColumn() << ",\"declarefilepath\":\""
//            << declareFilePath << "\",\"stmt\":\""
//            << escapeJsonString(Text.str()) << "\"}";
//       }
//       break;
//     case Stmt::CStyleCastExprClass:
//       ss << "{\"type\":\"typecast\",\"file\":\"" << filePath
//          << "\",\"line\":" << PLoc.getLine()
//          << ",\"column\":" << PLoc.getColumn() << ",\"declarefilepath\":\""
//          << declareFilePath << "\",\"stmt\":\"" <<
//          escapeJsonString(Text.str())
//          << "\"}";
//       break;
//     default:
//       break;
//     }
//     printf("%s\n", ss.str().c_str());
//     return true;
//   }

//   bool VisitTypeDecl(TypeDecl *TD) {
//     auto getFullNamespace = [](const clang::TypeDecl *typeDecl) {
//       std::vector<std::string> namespaceParts;
//       const clang::DeclContext *context = typeDecl->getDeclContext();
//       while (context && !context->isTranslationUnit()) {
//         if (const auto *ns = llvm::dyn_cast<clang::NamespaceDecl>(context)) {
//           if (!ns->isAnonymousNamespace()) {
//             namespaceParts.push_back(ns->getNameAsString());
//           }
//         } else if (const auto *record =
//                        llvm::dyn_cast<clang::CXXRecordDecl>(context)) {
//           namespaceParts.push_back(record->getNameAsString());
//         }
//         context = context->getParent();
//       }
//       std::reverse(namespaceParts.begin(), namespaceParts.end());
//       return llvm::join(namespaceParts, "::");
//     };

//     if (option.find("TYPE|") == std::string::npos)
//       return true;
//     std::string typeName = TD->getNameAsString();
//     if (typeName.empty())
//       return true;
//     typeName = getFullNamespace(TD) + "::" + typeName;
//     SourceLocation loc = TD->getLocation();
//     if (loc.isInvalid() || SM.isInSystemHeader(loc))
//       return true;
//     PresumedLoc PLoc = SM.getPresumedLoc(loc);
//     auto filePath = getPath(PLoc.getFilename());
//     if (skipPath(filePath))
//       return true;
//     std::stringstream ss;
//     ss << "{\"type\":\"typedecl\",\"file\":\"" << filePath
//        << "\",\"line\":" << PLoc.getLine() << ",\"column\":" <<
//        PLoc.getColumn()
//        << ",\"typename\":" << typeName << "\"}";
//     printf("%s\n", ss.str().c_str());
//     return true;
//   }

//   // bool VisitStmt(Stmt *S) {
//   //   SourceLocation Loc = SM.getSpellingLoc(S->getBeginLoc());
//   //   if (Loc.isInvalid() || SM.isInSystemHeader(Loc) ||
//   //   !SM.isWrittenInMainFile(Loc)) return true; PresumedLoc PLoc =
//   //   SM.getPresumedLoc(Loc); auto filePath = getPath(PLoc.getFilename());
//   if
//   //   (skipPath(filePath)) return true; SourceLocation BeginLoc =
//   //   Lexer::getLocForEndOfToken(S->getBeginLoc(), 0, SM, LangOptions());
//   //   SourceLocation EndLoc = Lexer::getLocForEndOfToken(S->getEndLoc(), 0,
//   SM,
//   //   LangOptions()); CharSourceRange FullRange =
//   //   CharSourceRange::getCharRange(BeginLoc, EndLoc); StringRef Text =
//   //   Lexer::getSourceText(FullRange, SM, LangOptions()); std::stringstream
//   ss;
//   //   ss << "stmt: " << S->getStmtClassName() << " line " << PLoc.getLine()
//   <<
//   //   " column " << PLoc.getColumn() << " text " << Text.str();
//   printf("%s\n",
//   //   ss.str().c_str()); return true;
//   // }

//   // bool VisitNestedNameSpecifierLoc(NestedNameSpecifierLoc NNSLoc) {
//   //   printf("VisitNestedNameSpecifierLoc\n");
//   //   NestedNameSpecifier *NNS = NNSLoc.getNestedNameSpecifier();
//   //   if (!NNS)
//   //     return true;

//   //   // 获取作用域类型和声明位置
//   //   std::string scopeType;
//   //   std::string scopeName;
//   //   SourceLocation declLoc;

//   //   // if (NNS->getKind() == NestedNameSpecifier::TypeSpec) {
//   //   //   QualType T = NNS->getAsType();
//   //   //   const Type *TP = T.getTypePtr();
//   //   //   if (TP->isRecordType()) {
//   //   //     CXXRecordDecl *RD = TP->getAsCXXRecordDecl();
//   //   //     if (RD) {
//   //   //       scopeType = "类";
//   //   //       scopeName = RD->getNameAsString();
//   //   //       declLoc = RD->getLocation();
//   //   //     }
//   //   //   }
//   //   // } else if (NNS->getKind() == NestedNameSpecifier::Namespace) {
//   //   //   NamespaceDecl *NS = NNS->getAsNamespace();
//   //   //   scopeType = "命名空间";
//   //   //   scopeName = NS->getNameAsString();
//   //   //   declLoc = NS->getLocation();
//   //   // } else if (NNS->getKind() == NestedNameSpecifier::Global) {
//   //   //   scopeType = "全局";
//   //   //   scopeName = "::";
//   //   //   // 全局作用域没有特定的Decl，可能需要特殊处理
//   //   // }

//   //   // // 获取声明位置的文件名和行号
//   //   // if (declLoc.isValid()) {
//   //   //   FullSourceLoc fullLoc = Context->getFullLoc(declLoc);
//   //   //   if (fullLoc.isValid()) {
//   //   //     llvm::outs() << "作用域类型: " << scopeType << ", 名称: " <<
//   //   scopeName
//   //   //                  << ", 声明位置: " <<
//   //   fullLoc.getFileEntry()->getName()
//   //   //                  << ":" << fullLoc.getSpellingLineNumber() << "\n";
//   //   //   }
//   //   // }

//   //   return true;
//   // }

//   // // 捕获命名空间定义
//   // bool VisitNamespaceDecl(NamespaceDecl *D) {
//   //   // report("NamespaceDef", D->getNameAsString(), D->getSourceRange());
//   //   SourceLocation Loc = SM.getSpellingLoc(D->getLocation());
//   //   if (Loc.isInvalid() || SM.isInSystemHeader(Loc) ||
//   //   !SM.isWrittenInMainFile(Loc)) return true; PresumedLoc PLoc =
//   //   SM.getPresumedLoc(Loc); std::stringstream ss; ss <<
//   //   "{\"type\":\"namespacedeclare\",\"file\":\"" << PLoc.getFilename() <<
//   //   "\",\"line\":" << PLoc.getLine() << ",\"column\":" << PLoc.getColumn()
//   <<
//   //   ",\"namespace\":\"" << D->getName().str() << "\"}"; printf("%s\n",
//   //   ss.str().c_str()); return true;
//   // }

//   // // 捕获using namespace声明
//   // bool VisitUsingDirectiveDecl(UsingDirectiveDecl *D) {
//   //   if (const NamespaceDecl *ND = D->getNominatedNamespace()) {
//   //     // report("UsingNamespace", ND->getNameAsString(),
//   //     D->getSourceRange()); SourceLocation Loc =
//   //     SM.getSpellingLoc(D->getLocation()); if (Loc.isInvalid() ||
//   //     SM.isInSystemHeader(Loc) || !SM.isWrittenInMainFile(Loc)) return
//   true;
//   //     PresumedLoc PLoc = SM.getPresumedLoc(Loc);
//   //     std::stringstream ss;
//   //     ss << "{\"type\":\"usingnamespace\",\"file\":\"" <<
//   PLoc.getFilename()
//   //     << "\",\"line\":" << PLoc.getLine() << ",\"column\":" <<
//   //     PLoc.getColumn() << "}"; printf("%s\n", ss.str().c_str());
//   //   }
//   //   return true;
//   // }

//   // // 捕获using声明
//   // bool VisitUsingDecl(UsingDecl *D) {
//   //   // std::string name;
//   //   // llvm::raw_string_ostream os(name);
//   //   // D->getQualifier()->print(os,
//   D->getASTContext().getPrintingPolicy());
//   //   // os << D->getNameAsString();
//   //   // report("UsingDecl", os.str(), D->getSourceRange());
//   //   SourceLocation Loc = SM.getSpellingLoc(D->getLocation());
//   //   if (Loc.isInvalid() || SM.isInSystemHeader(Loc) ||
//   //   !SM.isWrittenInMainFile(Loc)) return true; PresumedLoc PLoc =
//   //   SM.getPresumedLoc(Loc); std::stringstream ss; ss <<
//   //   "{\"type\":\"using\",\"file\":\"" << PLoc.getFilename() <<
//   "\",\"line\":"
//   //   << PLoc.getLine() << ",\"column\":" << PLoc.getColumn() << "}";
//   //   printf("%s\n", ss.str().c_str());
//   //   return true;
//   // }

//   // // 捕获函数定义
//   // bool VisitFunctionDecl(FunctionDecl *FD) {
//   //   // if (!FD->isThisDeclarationADefinition()) return true;     //
//   跳过声明
//   //   // if (SM.isInSystemHeader(FD->getLocation())) return true;  //
//   //   过滤系统函数
//   //   // analyzeFunction(FD);
//   //   SourceLocation Loc = SM.getSpellingLoc(FD->getLocation());
//   //   if (Loc.isInvalid() || SM.isInSystemHeader(Loc) ||
//   //   !SM.isWrittenInMainFile(Loc)) return true; PresumedLoc PLoc =
//   //   SM.getPresumedLoc(Loc); const CompoundStmt *Body =
//   //   dyn_cast_or_null<CompoundStmt>(FD->getBody()); std::stringstream ss;
//   if
//   //   (Body != nullptr) {
//   //     PresumedLoc PLocImp = SM.getPresumedLoc(Body->getLBracLoc());
//   //     ss << "{\"type\":\"functionimplement\",\"file\":\"" <<
//   //     PLocImp.getFilename() << "\",\"line\":" << PLocImp.getLine() <<
//   //     ",\"startcolumn\":" << PLoc.getColumn()  << ",\"endcolumn\":" <<
//   //     PLocImp.getColumn()-1 << ",\"functionname\":\"" <<
//   //     FD->getNameAsString() << "\"}";
//   //   } else {
//   //     ss << "{\"type\":\"functiondefine\",\"file\":\"" <<
//   PLoc.getFilename()
//   //     << "\",\"line\":" << PLoc.getLine() << ",\"column\":" <<
//   //     PLoc.getColumn() << ",\"functionname\":\"" << FD->getNameAsString()
//   <<
//   //     "\"}";
//   //   }
//   //   printf("%s\n", ss.str().c_str());
//   //   return true;
//   // }

//   // // 捕获普通函数调用
//   // bool VisitCallExpr(CallExpr *CE) {
//   //   if (FunctionDecl *FD = CE->getDirectCallee()) {
//   //     // reportCall(FD, CE->getSourceRange());
//   //     SourceLocation Loc =
//   //     SM.getSpellingLoc(CE->getSourceRange().getBegin()); SourceLocation
//   //     FDLoc = SM.getSpellingLoc(FD->getLocation()); if (FDLoc.isInvalid()
//   ||
//   //     SM.isInSystemHeader(FDLoc)) return true; PresumedLoc PLoc =
//   //     SM.getPresumedLoc(Loc); PresumedLoc PFDLoc =
//   SM.getPresumedLoc(FDLoc);
//   //     SourceLocation EndLoc =
//   //     Lexer::getLocForEndOfToken(CE->getSourceRange().getEnd(), 0, SM,
//   //     LangOptions()); CharSourceRange FullRange =
//   //     CharSourceRange::getCharRange(CE->getSourceRange().getBegin(),
//   EndLoc);
//   //     StringRef Text = Lexer::getSourceText(FullRange, SM, LangOptions());
//   //     std::stringstream ss;
//   //     ss << "{\"type\":\"functioncall\",\"file\":\"" << PLoc.getFilename()
//   <<
//   //     "\",\"line\":" << PLoc.getLine() << ",\"column\":" <<
//   PLoc.getColumn()
//   //     << ",\"functiondefinefile\":\"" << PFDLoc.getFilename() <<
//   //     "\",\"stmt\":\"" << Text.str() << "\"}"; printf("%s\n",
//   //     ss.str().c_str());
//   //   }
//   //   return true;
//   // }

//   // // 捕获成员函数调用
//   // bool VisitCXXMemberCallExpr(CXXMemberCallExpr *MCE) {
//   //   if (CXXMethodDecl *MD = MCE->getMethodDecl()) {
//   //     // reportCall(MD, MCE->getSourceRange());
//   //     SourceLocation Loc =
//   //     SM.getSpellingLoc(MCE->getSourceRange().getBegin()); SourceLocation
//   //     FDLoc = SM.getSpellingLoc(MD->getLocation()); if (FDLoc.isInvalid()
//   ||
//   //     SM.isInSystemHeader(FDLoc)) return true; PresumedLoc PLoc =
//   //     SM.getPresumedLoc(Loc); PresumedLoc PFDLoc =
//   SM.getPresumedLoc(FDLoc);
//   //     std::stringstream ss;
//   //     ss << "{\"type\":\"memberfunctioncall\",\"file\":\"" <<
//   //     PLoc.getFilename() << "\",\"line\":" << PLoc.getLine() <<
//   //     ",\"column\":" << PLoc.getColumn() << ",\"functiondefinefile\":\""
//   <<
//   //     PFDLoc.getFilename() << "\"}"; printf("%s\n", ss.str().c_str());
//   //   }
//   //   return true;
//   // }

//   // // 捕获类/结构体/联合体声明
//   // bool VisitCXXRecordDecl(CXXRecordDecl *D) {
//   //   SourceLocation Loc = SM.getSpellingLoc(D->getLocation());
//   //   if (Loc.isInvalid() || SM.isInSystemHeader(Loc)) return true;
//   //   PresumedLoc PLoc = SM.getPresumedLoc(Loc);
//   //   std::stringstream ss;
//   //   ss << "{\"type\":\"classdefine\",\"file\":\"" << PLoc.getFilename() <<
//   //   "\",\"line\":" << PLoc.getLine() << ",\"column\":" << PLoc.getColumn()
//   <<
//   //   ",\"definename\":\"" << D->getNameAsString() << "\"}"; printf("%s\n",
//   //   ss.str().c_str()); return true;
//   // }

//   // // 捕获全局变量、局部变量、静态变量
//   // bool VisitVarDecl(VarDecl *VD) {
//   //   SourceLocation Loc = SM.getSpellingLoc(VD->getLocation());
//   //   if (Loc.isInvalid() || SM.isInSystemHeader(Loc)) return true;
//   //   QualType Type = VD->getType();
//   //   std::string typeName = Type.getAsString();
//   //   PresumedLoc PLoc = SM.getPresumedLoc(Loc);
//   //   std::stringstream ss;
//   //   ss << "{\"type\":\"variabledefine\",\"file\":\"" << PLoc.getFilename()
//   <<
//   //   "\",\"line\":" << PLoc.getLine() << ",\"column\":" << PLoc.getColumn()
//   <<
//   //   ",\"definetype\":\"" << typeName << "\"}"; printf("%s\n",
//   //   ss.str().c_str()); return true;
//   // }

//   // // 捕获类成员变量
//   // bool VisitFieldDecl(FieldDecl *FD) {
//   //   SourceLocation Loc = SM.getSpellingLoc(FD->getLocation());
//   //   if (Loc.isInvalid() || SM.isInSystemHeader(Loc)) return true;
//   //   QualType Type = FD->getType();
//   //   std::string typeName = Type.getAsString();
//   //   PresumedLoc PLoc = SM.getPresumedLoc(Loc);
//   //   std::stringstream ss;
//   //   ss << "{\"type\":\"fielddefine\",\"file\":\"" << PLoc.getFilename() <<
//   //   "\",\"line\":" << PLoc.getLine() << ",\"column\":" << PLoc.getColumn()
//   <<
//   //   ",\"definetype\":\"" << typeName << "\"}"; printf("%s\n",
//   //   ss.str().c_str()); return true;
//   // }

//   // // 捕获函数参数
//   // bool VisitParmVarDecl(ParmVarDecl *PD) {
//   //   SourceLocation Loc = SM.getSpellingLoc(PD->getLocation());
//   //   if (Loc.isInvalid() || SM.isInSystemHeader(Loc)) return true;
//   //   QualType Type = PD->getType();
//   //   std::string typeName = Type.getAsString();
//   //   PresumedLoc PLoc = SM.getPresumedLoc(Loc);
//   //   std::stringstream ss;
//   //   ss << "{\"type\":\"parameterdefine\",\"file\":\"" <<
//   PLoc.getFilename()
//   //   << "\",\"line\":" << PLoc.getLine() << ",\"column\":" <<
//   PLoc.getColumn()
//   //   << ",\"definetype\":\"" << typeName << "\"}"; printf("%s\n",
//   //   ss.str().c_str()); return true;
//   // }

// private:
//   ASTContext &Context;
//   SourceManager &SM;
// };

// // AST消费者类
// class AnalysisConsumer : public ASTConsumer {
// public:
//   explicit AnalysisConsumer(ASTContext &Context, SourceManager &SM)
//       : Visitor(Context, SM) {}
//   void HandleTranslationUnit(ASTContext &Context) override {
//     Visitor.TraverseDecl(Context.getTranslationUnitDecl());
//   }
// private:
//   AnalysisVisitor Visitor;
// };

// // FrontendAction类
// class AnalysisAction : public ASTFrontendAction {
// public:
//   std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
//                                                  StringRef File) override {
//     return std::make_unique<AnalysisConsumer>(CI.getASTContext(),
//                                               CI.getSourceManager());
//   }
// };
