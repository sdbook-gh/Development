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
