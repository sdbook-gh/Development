#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/MacroArgs.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include <sstream>

using namespace clang;
using namespace clang::tooling;

class MacroAnalysisCallback : public PPCallbacks {
  Preprocessor &PP;

 public:
  MacroAnalysisCallback(Preprocessor &PP) : PP(PP) {}

  void MacroDefined(const Token &MacroNameTok,
                    const MacroDirective *MD) override {
    const MacroInfo *MI = MD->getMacroInfo();
    SourceManager &SM = PP.getSourceManager();
    SourceLocation Loc = SM.getSpellingLoc(MacroNameTok.getLocation());
    if (Loc.isInvalid() || SM.isInSystemHeader(Loc)) return;
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    std::string filename{PLoc.getFilename() != nullptr ? PLoc.getFilename() : ""};
    if (filename.empty() || filename.find("<") != std::string::npos) return;
    // StringRef MacroText = PP.getMacroSourceText(MI, SM);
    std::stringstream ss;
    ss << "{type:macrodefine,file:" << PLoc.getFilename() << + ",line:" << PLoc.getLine() << ",cloumn:" << PLoc.getColumn()<< ",info:" << MacroNameTok.getIdentifierInfo()->getName().str() << "}";
    printf("%s\n", ss.str().c_str());
  }

  void MacroExpands(const Token &MacroNameTok,
                    const MacroDefinition &MacroDefinition, SourceRange Range,
                    const MacroArgs *Args) override {
    SourceManager &SM = PP.getSourceManager();
    SourceLocation Loc = SM.getSpellingLoc(MacroNameTok.getLocation());
    if (Loc.isInvalid() || SM.isInSystemHeader(Loc) || !SM.isWrittenInMainFile(Loc)) return;
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    std::stringstream ss;
    ss << "{type:macroexpand,file:" << PLoc.getFilename() << + ",line:" << PLoc.getLine() << ",cloumn:" << PLoc.getColumn()<< ",info:" << MacroNameTok.getIdentifierInfo()->getName().str() << "}";
    printf("%s\n", ss.str().c_str());
  }

};

class MacroAnalysisAction : public PreprocessorFrontendAction {
 protected:
  void ExecuteAction() override {
    Preprocessor &PP = getCompilerInstance().getPreprocessor();
    PP.addPPCallbacks(std::make_unique<MacroAnalysisCallback>(PP));

    // Process all tokens to trigger preprocessing callbacks
    PP.EnterMainSourceFile();
    Token Tok;
    do {
      PP.Lex(Tok);
    } while (Tok.isNot(tok::eof));
  }
};

#include <clang/AST/ASTConsumer.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendAction.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/Support/CommandLine.h>

using namespace clang;
using namespace clang::tooling;

// 定义命令行选项
static llvm::cl::OptionCategory NsCategory("Namespace Analysis Options");

// AST访问器类
class NamespaceVisitor : public RecursiveASTVisitor<NamespaceVisitor> {
 public:
  explicit NamespaceVisitor(ASTContext &Context, SourceManager &SM)
      : Context(Context), SM(SM) {}

  // 捕获命名空间定义
  bool VisitNamespaceDecl(NamespaceDecl *D) {
    // report("NamespaceDef", D->getNameAsString(), D->getSourceRange());
    SourceLocation Loc = SM.getSpellingLoc(D->getLocation());
    if (Loc.isInvalid() || SM.isInSystemHeader(Loc) || !SM.isWrittenInMainFile(Loc)) return true;
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    std::stringstream ss;
    ss << "{type:namespacedeclare,file:" << PLoc.getFilename() << + ",line:" << PLoc.getLine() << ",cloumn:" << PLoc.getColumn()<< ",info:" << "null" << "}";
    printf("%s\n", ss.str().c_str());
    return true;
  }

  // 捕获using namespace声明
  bool VisitUsingDirectiveDecl(UsingDirectiveDecl *D) {
    if (const NamespaceDecl *ND = D->getNominatedNamespace()) {
      // report("UsingNamespace", ND->getNameAsString(), D->getSourceRange());
      SourceLocation Loc = SM.getSpellingLoc(D->getLocation());
      if (Loc.isInvalid() || SM.isInSystemHeader(Loc) || !SM.isWrittenInMainFile(Loc)) return true;
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      std::stringstream ss;
      ss << "{type:usingnamespace,file:" << PLoc.getFilename() << + ",line:" << PLoc.getLine() << ",cloumn:" << PLoc.getColumn()<< ",info:" << "null" << "}";
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
    if (Loc.isInvalid() || SM.isInSystemHeader(Loc) || !SM.isWrittenInMainFile(Loc)) return true;
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    std::stringstream ss;
    ss << "{type:using,file:" << PLoc.getFilename() << + ",line:" << PLoc.getLine() << ",cloumn:" << PLoc.getColumn()<< ",info:" << "null" << "}";
    printf("%s\n", ss.str().c_str());
    return true;
  }

  // 捕获函数定义
  bool VisitFunctionDecl(FunctionDecl *FD) {
    // if (!FD->isThisDeclarationADefinition()) return true;     // 跳过声明
    // if (SM.isInSystemHeader(FD->getLocation())) return true;  // 过滤系统函数
    // analyzeFunction(FD);
    SourceLocation Loc = SM.getSpellingLoc(FD->getLocation());
    if (Loc.isInvalid() || SM.isInSystemHeader(Loc) || !SM.isWrittenInMainFile(Loc)) return true;
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    std::stringstream ss;
    ss << "{type:functiondefine,file:" << PLoc.getFilename() << + ",line:" << PLoc.getLine() << ",cloumn:" << PLoc.getColumn()<< ",info:" << FD->getNameAsString() << "}";
    printf("%s\n", ss.str().c_str());
    return true;
  }

  // 捕获普通函数调用
  bool VisitCallExpr(CallExpr *CE) {
    if (FunctionDecl *FD = CE->getDirectCallee()) {
      // reportCall(FD, CE->getSourceRange());
      SourceLocation Loc = SM.getSpellingLoc(CE->getSourceRange().getBegin());
      SourceLocation FDLoc = SM.getSpellingLoc(FD->getLocation());
      if (FDLoc.isInvalid() || SM.isInSystemHeader(FDLoc)) return true;
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      std::stringstream ss;
      ss << "{type:functioncall,file:" << PLoc.getFilename() << + ",line:" << PLoc.getLine() << ",cloumn:" << PLoc.getColumn()<< ",info:" << FD->getNameAsString() << "}";
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
      if (FDLoc.isInvalid() || SM.isInSystemHeader(FDLoc)) return true;
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      std::stringstream ss;
      ss << "{type:memberfunctioncall,file:" << PLoc.getFilename() << + ",line:" << PLoc.getLine() << ",cloumn:" << PLoc.getColumn()<< ",info:" << << MD->getNameAsString() << "}";
      printf("%s\n", ss.str().c_str());
    }
    return true;
  }

  // 捕获类/结构体/联合体声明
  bool VisitCXXRecordDecl(CXXRecordDecl *D) {
    SourceLocation Loc = SM.getSpellingLoc(D->getLocation());
    if (Loc.isInvalid() || SM.isInSystemHeader(Loc)) return true;
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    std::stringstream ss;
    ss << "{type:classdefine,file:" << PLoc.getFilename() << + ",line:" << PLoc.getLine() << ",cloumn:" << PLoc.getColumn()<< ",info:" << D->getNameAsString() << "}";
    printf("%s\n", ss.str().c_str());
    return true;
  }

  // 捕获全局变量、局部变量、静态变量
  bool VisitVarDecl(VarDecl *VD) {
    SourceLocation Loc = SM.getSpellingLoc(VD->getLocation());
    if (Loc.isInvalid() || SM.isInSystemHeader(Loc)) return true;
    QualType Type = VD->getType();
    std::string typeName = Type.getAsString();
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    std::stringstream ss;
    ss << "{type:variabledefine,file:" << PLoc.getFilename() << + ",line:" << PLoc.getLine() << ",cloumn:" << PLoc.getColumn()<< ",info:" << typeName << "}";
    printf("%s\n", ss.str().c_str());
    return true;
  }

  // 捕获类成员变量
  bool VisitFieldDecl(FieldDecl *FD) {
    SourceLocation Loc = SM.getSpellingLoc(FD->getLocation());
    if (Loc.isInvalid() || SM.isInSystemHeader(Loc)) return true;
    QualType Type = FD->getType();
    std::string typeName = Type.getAsString();
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    std::stringstream ss;
    ss << "{type:fielddefine,file:" << PLoc.getFilename() << + ",line:" << PLoc.getLine() << ",cloumn:" << PLoc.getColumn()<< ",info:" << typeName << "}";
    printf("%s\n", ss.str().c_str());
    return true;
  }

  // 捕获函数参数
  bool VisitParmVarDecl(ParmVarDecl *PD) {
    SourceLocation Loc = SM.getSpellingLoc(PD->getLocation());
    if (Loc.isInvalid() || SM.isInSystemHeader(Loc)) return true;
    QualType Type = PD->getType();
    std::string typeName = Type.getAsString();
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    std::stringstream ss;
    ss << "{type:parameterdefine,file:" << PLoc.getFilename() << + ",line:" << PLoc.getLine() << ",cloumn:" << PLoc.getColumn()<< ",info:" << typeName << "}";
    printf("%s\n", ss.str().c_str());
    return true;
  }

 private:
  // std::string getSourceLocationString(Preprocessor &PP, SourceLocation Loc) {
  //   if (Loc.isInvalid()) return std::string("(none)");
  //   if (Loc.isFileID()) {
  //     PresumedLoc PLoc = PP.getSourceManager().getPresumedLoc(Loc);
  //     if (PLoc.isInvalid()) {
  //       return std::string("(invalid)");
  //     }
  //     std::string Str;
  //     llvm::raw_string_ostream SS(Str);
  //     // The macro expansion and spelling pos is identical for file locs.
  //     SS << "\"" << PLoc.getFilename() << ':' << PLoc.getLine() << ':'
  //        << PLoc.getColumn() << "\"";
  //     std::string Result = SS.str();
  //     return Result;
  //   }
  //   return std::string("(nonfile)");
  // }

  // std::string getInfo(Preprocessor &PP, SourceRange Value) {
  //   if (Value.isInvalid()) {
  //     return "";
  //   }
  //   std::string Str;
  //   llvm::raw_string_ostream SS(Str);
  //   SS << "[" << getSourceLocationString(PP, Value.getBegin()) << ", "
  //      << getSourceLocationString(PP, Value.getEnd()) << "]";
  //   std::string Result = SS.str();
  //   return Result;
  // }

  // std::string getInfo(Preprocessor &PP, const MacroArgs *Value) {
  //   if (!Value) {
  //     return "";
  //   }
  //   std::string Str;
  //   llvm::raw_string_ostream SS(Str);
  //   SS << "[";

  //   // Each argument is is a series of contiguous Tokens, terminated by a eof.
  //   // Go through each argument printing tokens until we reach eof.
  //   for (unsigned I = 0; I < Value->getNumMacroArguments(); ++I) {
  //     const Token *Current = Value->getUnexpArgument(I);
  //     if (I) SS << ", ";
  //     bool First = true;
  //     while (Current->isNot(tok::eof)) {
  //       if (!First) SS << " ";
  //       // We need to be careful here because the arguments might not be legal
  //       // in YAML, so we use the token name for anything but identifiers and
  //       // numeric literals.
  //       if (Current->isAnyIdentifier() || Current->is(tok::numeric_constant)) {
  //         SS << PP.getSpelling(*Current);
  //       } else {
  //         SS << "<" << Current->getName() << ">";
  //       }
  //       ++Current;
  //       First = false;
  //     }
  //   }
  //   SS << "]";
  //   std::string Result = SS.str();
  //   return Result;
  // }

  // void report(const char *Type, const std::string &Name, SourceRange Range) {
  //   // 获取开始和结束位置
  //   SourceLocation BeginLoc = Range.getBegin();
  //   SourceLocation EndLoc = Range.getEnd();
  //   // 过滤无效和系统头文件位置
  //   if (BeginLoc.isInvalid() || SM.isInSystemHeader(BeginLoc)) return;
  //   if (EndLoc.isInvalid() || SM.isInSystemHeader(EndLoc)) return;
  //   // 转换为可读位置
  //   PresumedLoc PLocBegin = SM.getPresumedLoc(BeginLoc);
  //   PresumedLoc PLocEnd = SM.getPresumedLoc(EndLoc);
  //   llvm::outs() << "[" << Type << "] " << Name << "\n"
  //                << "  Start: " << PLocBegin.getFilename() << ":"
  //                << PLocBegin.getLine() << ":" << PLocBegin.getColumn() << "\n"
  //                << "  End:   " << PLocEnd.getFilename() << ":"
  //                << PLocEnd.getLine() << ":" << PLocEnd.getColumn() << "\n\n";
  // }
  // // 辅助函数：获取完整的命名空间链
  // std::string getNamespaceChain(const DeclContext *DC) {
  //   std::vector<std::string> namespaces;
  //   while (DC && !DC->isTranslationUnit()) {
  //     if (const auto *NS = dyn_cast<NamespaceDecl>(DC)) {
  //       if (!NS->isAnonymousNamespace()) {
  //         namespaces.push_back(NS->getNameAsString());
  //       }
  //     }
  //     DC = DC->getParent();
  //   }
  //   std::reverse(namespaces.begin(), namespaces.end());
  //   return llvm::join(namespaces, "::");
  // }
  // std::string getLocationString(SourceLocation Loc, SourceManager &SM) {
  //   if (Loc.isInvalid() || SM.isInSystemHeader(Loc)) return "";
  //   PresumedLoc PLoc = SM.getPresumedLoc(Loc);
  //   return std::string(PLoc.getFilename()) + ":" +
  //          std::to_string(PLoc.getLine()) + ":" +
  //          std::to_string(PLoc.getColumn());
  // }
  // void analyzeFunction(FunctionDecl *FD) {
  //   // 基础信息
  //   std::string funcName = FD->getNameAsString();
  //   std::string returnType = FD->getReturnType().getAsString();
  //   // 参数信息
  //   std::vector<std::string> params;
  //   for (ParmVarDecl *P : FD->parameters()) {
  //     params.push_back(P->getType().getAsString() + " " + P->getNameAsString());
  //   }
  //   // 位置信息
  //   SourceRange funcRange = FD->getSourceRange();
  //   // std::string locStart = getLocationString(funcRange.getBegin());
  //   // std::string locEnd = getLocationString(funcRange.getEnd());
  //   std::string loc = getLocationString(FD->getLocation(), SM);
  //   // 输出基本信息
  //   llvm::outs() << "Function Implementation:\n"
  //                << "Name: " << returnType << " " << funcName << "("
  //                << llvm::join(params, ", ") << ")\n"
  //                << "Location: " << loc << "\n";

  //   // 分析函数体
  //   if (Stmt *Body = FD->getBody()) {
  //     // analyzeFunctionBody(Body);
  //   }
  //   llvm::outs() << "-----------------------\n";
  // }
  // void reportCall(FunctionDecl *FD, SourceRange Range) {
  //   // 获取命名空间信息
  //   std::string nsChain = getNamespaceChain(FD->getDeclContext());
  //   // 获取完整函数签名
  //   std::string funcName = nsChain.empty()
  //                              ? FD->getNameAsString()
  //                              : nsChain + "::" + FD->getNameAsString();
  //   std::string returnType = FD->getReturnType().getAsString();
  //   // 获取参数列表
  //   std::vector<std::string> params;
  //   for (auto *P : FD->parameters()) {
  //     params.push_back(P->getType().getAsString() + " " + P->getNameAsString());
  //   }
  //   // 获取声明位置信息
  //   const std::string declLoc = getLocationString(FD->getLocation(), SM);
  //   // 获取函数签名
  //   std::string funcSig;
  //   llvm::raw_string_ostream os(funcSig);
  //   FD->printQualifiedName(os);
  //   os << FD->getType().getAsString();
  //   // 获取位置信息
  //   SourceLocation Begin = Range.getBegin();
  //   SourceLocation End = Range.getEnd();
  //   // if (Begin.isInvalid() || SM.isInSystemHeader(Begin)) return;
  //   PresumedLoc PLocBegin = SM.getPresumedLoc(Begin);
  //   PresumedLoc PLocEnd = SM.getPresumedLoc(End);
  //   // if (nsChain.empty() || nsChain.find("std::") != std::string::npos ||
  //   // nsChain == "std") {
  //   //   return;
  //   // }
  //   // 打印结果
  //   llvm::outs() << "Function Call:\n"
  //                << "  Declaration: " << os.str() << "\n"
  //                << "  Declared At: "
  //                << (declLoc.empty() ? "<built-in>" : declLoc) << "\n"
  //                << "  Name: " << funcName << "\n"
  //                << "  Namespace: " << (nsChain.empty() ? "(global)" : nsChain)
  //                << "\n"
  //                << "  Returntype:" << returnType << "\n"
  //                << "  Parameters: (" << llvm::join(params, ", ") << ")\n"
  //                << "  Location: " << PLocBegin.getFilename() << " ["
  //                << PLocBegin.getLine() << ":" << PLocBegin.getColumn() << " - "
  //                << PLocEnd.getLine() << ":" << PLocEnd.getColumn() << "]\n\n";
  // }

  ASTContext &Context;
  SourceManager &SM;
};

// AST消费者类
class NamespaceConsumer : public ASTConsumer {
 public:
  explicit NamespaceConsumer(ASTContext &Context, SourceManager &SM)
      : Visitor(Context, SM) {}

  void HandleTranslationUnit(ASTContext &Context) override {
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }

 private:
  NamespaceVisitor Visitor;
};

// FrontendAction类
class NamespaceAction : public ASTFrontendAction {
 public:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef File) override {
    return std::make_unique<NamespaceConsumer>(CI.getASTContext(),
                                               CI.getSourceManager());
  }
};

int main(int argc, const char **argv) {
  // 解析命令行参数和编译命令
  auto ExpectedParser = CommonOptionsParser::create(argc, argv, NsCategory);
  if (!ExpectedParser) {
    llvm::errs() << llvm::toString(ExpectedParser.takeError());
    return 1;
  }

  // 创建工具并运行
  ClangTool Tool(ExpectedParser->getCompilations(),
                 ExpectedParser->getSourcePathList());
  Tool.run(newFrontendActionFactory<MacroAnalysisAction>().get());
  Tool.run(newFrontendActionFactory<NamespaceAction>().get());
  return 0;
}
