#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang;
using namespace clang::tooling;
using namespace clang::ast_matchers;

class MyMatchCallback : public MatchFinder::MatchCallback {
public:
    virtual void run(const MatchFinder::MatchResult &Result) {
        if (const CXXRecordDecl *CRD = Result.Nodes.getNodeAs<CXXRecordDecl>("AMapLibDecl")) {
          auto& SM = Result.Context->getSourceManager();
          SourceLocation Loc = SM.getSpellingLoc(CRD->getLocation());
          if (Loc.isInvalid() || SM.isInSystemHeader(Loc)) return;
          PresumedLoc PLoc = SM.getPresumedLoc(Loc);
          std::stringstream ss;
          ss << "{\"type\":\"classdefine\",\"file\":\"" << PLoc.getFilename() << "\",\"line\":" << PLoc.getLine() << ",\"column\":" << PLoc.getColumn() << ",\"definename\":\"" << CRD->getNameAsString() << "\"}";
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
    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef file) override {
        MatchFinder Finder;
        auto ClassMatcher = cxxRecordDecl(hasName("AMapLib")).bind("AMapLibDecl");
        Finder.addMatcher(ClassMatcher, new MyMatchCallback());
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
