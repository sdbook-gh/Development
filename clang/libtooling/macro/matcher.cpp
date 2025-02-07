#include "clang/AST/Expr.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/TypeLoc.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"

#include <sstream>

using namespace clang;
using namespace clang::tooling;
using namespace clang::ast_matchers;

class MyMatchCallback : public MatchFinder::MatchCallback {
public:
  virtual void run(const MatchFinder::MatchResult &Result) {
    if (const auto *TL = Result.Nodes.getNodeAs<TypeLoc>("checktype")) {
      auto &SM = Result.Context->getSourceManager();
      SourceLocation Loc = SM.getSpellingLoc(TL->getBeginLoc());
      if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
        return;
      SourceLocation DLoc = Loc;
      if (auto ETL = TL->getAs<ElaboratedTypeLoc>()) {
        auto Q = ETL.getQualifierLoc();
        DLoc = SM.getSpellingLoc(Q.getBeginLoc());
      } else if (auto RTL = TL->getAs<RecordTypeLoc>()) {
        auto *D = RTL.getDecl();
        DLoc = SM.getSpellingLoc(D->getBeginLoc());
      }
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      PresumedLoc DPLoc = PLoc;
      if (auto RTL = TL->getAs<RecordTypeLoc>()) {
        DPLoc = SM.getPresumedLoc(DLoc);
      }
      std::stringstream ss;
      ss << "{\"type\":\"TypeLoc\",\"file\":\"" << PLoc.getFilename()
         << "\",\"line\":" << PLoc.getLine()
         << ",\"column\":" << PLoc.getColumn() << ",\"definename\":\""
         << TL->getType()->getTypeClassName() << "\",\"dfile\":\""
         << DPLoc.getFilename() << "\",\"dline\":" << DPLoc.getLine()
         << ",\"dcolumn\":" << DPLoc.getColumn() << "\"}";
      printf("%s\n", ss.str().c_str());
    }
    if (const auto *DRE = Result.Nodes.getNodeAs<DeclRefExpr>("checkdeclref")) {
      auto &SM = Result.Context->getSourceManager();
      SourceLocation Loc = SM.getSpellingLoc(DRE->getBeginLoc());
      if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
        return;
      const auto *D = DRE->getDecl();
      auto DLoc = SM.getSpellingLoc(D->getBeginLoc());
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      auto DPLoc = SM.getPresumedLoc(DLoc);
      std::stringstream ss;
      ss << "{\"type\":\"DeclRefExpr\",\"file\":\"" << PLoc.getFilename()
         << "\",\"line\":" << PLoc.getLine()
         << ",\"column\":" << PLoc.getColumn() << ",\"definename\":\""
         << DRE->getType()->getTypeClassName() << "\",\"dfile\":\""
         << DPLoc.getFilename() << "\",\"dline\":" << DPLoc.getLine()
         << ",\"dcolumn\":" << DPLoc.getColumn() << "\"}";
      printf("%s\n", ss.str().c_str());
    }
  }
};

class MyASTConsumer : public ASTConsumer {
public:
  explicit MyASTConsumer(MatchFinder &Finder) : Finder(Finder) {}

  void HandleTranslationUnit(ASTContext &Context) override {
    Finder.matchAST(Context);
  }

private:
  MatchFinder Finder;
};

class MyFrontendAction : public ASTFrontendAction {
public:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef file) override {
    MatchFinder Finder;
    auto *MyCallback{new MyMatchCallback};
    Finder.addMatcher(clang::ast_matchers::typeLoc().bind("checktype"),
                      MyCallback);
    Finder.addMatcher(clang::ast_matchers::declRefExpr().bind("checkdeclref"),
                      MyCallback);
    return std::make_unique<MyASTConsumer>(Finder);
  }
};

int main(int argc, const char **argv) {
  llvm::cl::OptionCategory Category("Analysis Options");
  auto ExpectedParser = CommonOptionsParser::create(argc, argv, Category);
  if (!ExpectedParser) {
    llvm::errs() << llvm::toString(ExpectedParser.takeError());
    return 1;
  }
  // 创建工具并运行
  ClangTool Tool(ExpectedParser->getCompilations(),
                 ExpectedParser->getSourcePathList());
  Tool.run(newFrontendActionFactory<MyFrontendAction>().get());
  return 0;
}
