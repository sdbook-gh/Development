#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclAccessPair.h"
#include "clang/AST/DeclCXX.h"
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
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

using namespace clang;
using namespace clang::tooling;
using namespace clang::ast_matchers;

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
  auto input_string = input;
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

static std::string option;

class AnalysisPPCallback : public PPCallbacks {
  Preprocessor &PP;

public:
  AnalysisPPCallback(Preprocessor &PP) : PP(PP) {}

  void MacroDefined(const Token &MacroNameTok,
                    const MacroDirective *MD) override {
    if (option.find("MacroDefExp|") == std::string::npos)
      return;
    const MacroInfo *MI = MD->getMacroInfo();
    SourceManager &SM = PP.getSourceManager();
    SourceLocation Loc = MacroNameTok.getLocation();
    if (SM.isInSystemHeader(Loc))
      return;
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    auto filePath = getPath(PLoc.getFilename());
    if (skipPath(filePath))
      return;
    SourceLocation BeginLoc =
        Lexer::GetBeginningOfToken(MI->getDefinitionLoc(), SM, LangOptions());
    SourceLocation EndLoc = Lexer::getLocForEndOfToken(
        MI->getDefinitionEndLoc(), 0, SM, LangOptions());
    CharSourceRange FullRange = CharSourceRange::getCharRange(BeginLoc, EndLoc);
    std::string name =
        escapeJsonString(MacroNameTok.getIdentifierInfo()->getName().str());
    std::string stmt = escapeJsonString(
        Lexer::getSourceText(FullRange, SM, LangOptions()).str());
    std::stringstream ss;
    ss << "{\"type\":\"MacroDefined\",\"file\":\"" << filePath
       << "\",\"line\":" << PLoc.getLine() << ",\"column\":" << PLoc.getColumn()
       << ",\"macroname\":\"" << name << "\",\"macrostmt\":\"" << stmt << "\"}";
    printf("%s\n", ss.str().c_str());
  }

  void MacroExpands(const Token &MacroNameTok,
                    const MacroDefinition &MacroDefinition, SourceRange Range,
                    const MacroArgs *Args) override {
    if (option.find("MacroDefExp|") == std::string::npos)
      return;
    SourceManager &SM = PP.getSourceManager();
    SourceLocation Loc = MacroNameTok.getLocation();
    if (SM.isInSystemHeader(Loc))
      return;
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    auto filePath = getPath(PLoc.getFilename());
    if (skipPath(filePath))
      return;
    std::string name =
        escapeJsonString(MacroNameTok.getIdentifierInfo()->getName().str());
    std::string stmt = name;
    std::stringstream ss;
    ss << "{\"type\":\"MacroExpands\",\"file\":\"" << filePath
       << "\",\"line\":" << PLoc.getLine() << ",\"column\":" << PLoc.getColumn()
       << ",\"macroname\":\"" << name << "\",\"macrostmt\":\"" << stmt << "\"}";
    printf("%s\n", ss.str().c_str());
  }

  //   void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
  //                           StringRef FileName, bool IsAngled,
  //                           CharSourceRange FilenameRange,
  //                           OptionalFileEntryRef File, StringRef SearchPath,
  //                           StringRef RelativePath, const Module *Imported,
  //                           SrcMgr::CharacteristicKind FileType) override {
  //     if (option.find("InclusionDirective|") == std::string::npos)
  //       return;
  //     SourceManager &SM = PP.getSourceManager();
  //     SourceLocation Loc = SM.getSpellingLoc(IncludeTok.getLocation());
  //     if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
  //       return;
  //     PresumedLoc PLoc = SM.getPresumedLoc(Loc);
  //     auto filePath = getPath(PLoc.getFilename());
  //     if (skipPath(filePath))
  //       return;
  //     std::string includeFilePath{getPath(File->getName().str())};
  //     if (skipPath(includeFilePath))
  //       return;
  //     SourceLocation EndLoc =
  //     Lexer::getLocForEndOfToken(FilenameRange.getEnd(),
  //                                                        0, SM,
  //                                                        LangOptions());
  //     CharSourceRange FullRange = CharSourceRange::getCharRange(HashLoc,
  //     EndLoc); StringRef Text = Lexer::getSourceText(FullRange, SM,
  //     LangOptions()); std::stringstream ss; ss <<
  //     "{\"type\":\"InclusionDirective\",\"file\":\"" << filePath
  //        << "\",\"line\":" << PLoc.getLine() << ",\"column\":" <<
  //        PLoc.getColumn()
  //        << ",\"includefilepath\":\"" << includeFilePath << "\",\"stmt\":\""
  //        << escapeJsonString(Text.str()) << "\"}";
  //     printf("%s\n", ss.str().c_str());
  //   }
};

class PPAnalysisAction : public PreprocessorFrontendAction {
protected:
  void ExecuteAction() override {
    Preprocessor &PP = getCompilerInstance().getPreprocessor();
    PP.addPPCallbacks(std::make_unique<AnalysisPPCallback>(PP));
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

  virtual void run(const MatchFinder::MatchResult &Result) {
    if (const NamespaceDecl *ND =
            Result.Nodes.getNodeAs<NamespaceDecl>("NamespaceDecl|")) {
      std::string file, Dfile, stmt, Dstmt;
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
        stmt = escapeJsonString(ND->getNameAsString());
      }
      if (skipPath(file))
        return;
      std::stringstream ss;
      ss << "{\"type\":\"NamespaceDecl\",\"file\":\"" << file
         << "\",\"line\":" << line << ",\"column\":" << column << ",\"stmt\":\""
         << stmt << "\",\"exptype\":\""
         << "NamespaceDecl"
         << "\",\"dfile\":\"" << Dfile << "\",\"dline\":" << Dline
         << ",\"dcolumn\":" << Dcolumn << ",\"dstmt\":\"" << Dstmt << "\"}";
      printf("%s\n", ss.str().c_str());
    } else if (const TypeLoc *TL =
                   Result.Nodes.getNodeAs<TypeLoc>("TypeLoc|")) {
// #define LOG
#ifdef LOG
      // printf("++++++\n");
#endif
      std::string file, Dfile, stmt, Dstmt;
      int line = 0, column = 0;
      int Dline = 0, Dcolumn = 0;
      static std::vector<std::string> stored_record_vec;
      SourceLocation Loc = SM.getSpellingLoc(TL->getBeginLoc());
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      if (PLoc.isValid()) {
        file = getPath(PLoc.getFilename());
        line = PLoc.getLine();
        column = PLoc.getColumn();
      }
      {
        CharSourceRange FullRange =
            CharSourceRange::getCharRange(TL->getBeginLoc(), TL->getEndLoc());
        StringRef Text = Lexer::getSourceText(FullRange, SM, LangOptions());
        stmt = escapeJsonString(Text.str());
      }
      SourceLocation DLoc = Loc;
      QualType QT = TL->getType();
      std::string exptype = QT->getTypeClassName();
      if (ElaboratedTypeLoc ETL = TL->getAs<ElaboratedTypeLoc>()) {
        NestedNameSpecifierLoc QL = ETL.getQualifierLoc();
        if (QL.hasQualifier()) {
          const NestedNameSpecifier *NNS = QL.getNestedNameSpecifier();
          if (NNS) {
            const Decl *D = nullptr;
            if (const Type *T = NNS->getAsType()) {
              if (const TagDecl *TD = T->getAsTagDecl()) {
                D = TD;
              }
            } else {
              D = NNS->getAsNamespace();
            }
            if (D) {
              DLoc = D->getLocation();
              // PresumedLoc DPLoc = SM.getPresumedLoc(DLoc);
              // if (DPLoc.isInvalid() || SM.isInSystemHeader(DLoc))
              //   return;
              // Dfile = getPath(DPLoc.getFilename());
              // Dline = DPLoc.getLine();
              // Dcolumn = DPLoc.getColumn();
              // // printf("++++++ %s\n", Dfile.c_str());
              // if (!Dfile.empty() && skipPath(Dfile))
              //   return;
            }
          }
        }
      } else if (FunctionProtoTypeLoc FPT = TL->getAs<FunctionProtoTypeLoc>()) {
        return;
      } else if (RecordTypeLoc RTL = TL->getAs<RecordTypeLoc>()) {
        RecordDecl *D = RTL.getDecl();
        DLoc = D->getBeginLoc();
        stmt = escapeJsonString(D->getNameAsString());
      } else if (TypedefTypeLoc TTL = TL->getAs<TypedefTypeLoc>()) {
        TypedefNameDecl *TND = TTL.getTypedefNameDecl();
        DLoc = TND->getBeginLoc();
        stmt = escapeJsonString(TND->getNameAsString());
      } else if (EnumTypeLoc ETL = TL->getAs<EnumTypeLoc>()) {
        // CharSourceRange FullRange =
        //     CharSourceRange::getTokenRange(TL->getBeginLoc(), TL->getEndLoc());
        // StringRef Text = Lexer::getSourceText(FullRange, SM, LangOptions());
        // stmt = escapeJsonString(Text.str());
        EnumDecl *D = ETL.getDecl();
        DLoc = D->getBeginLoc();
        stmt = escapeJsonString(D->getNameAsString());
        // stmt = escapeJsonString(D->getQualifiedNameAsString());
      } else if (PointerTypeLoc PTL = TL->getAs<PointerTypeLoc>()) {
        return;
        // QualType QT = PTL.getType();
        // const Type *T = QT.getTypePtr();
        // QT = T->getPointeeType();
        // const Type *TP = QT.getTypePtr();
        // if (TagDecl *TD = TP->getAsTagDecl()) {
        //   DLoc = TD->getLocation();
        // }
      } else if (ReferenceTypeLoc RTL = TL->getAs<ReferenceTypeLoc>()) {
        return;
        // QualType QT = RTL.getType();
        // const Type *T = QT.getTypePtr();
        // QT = T->getPointeeType();
        // const Type *TP = QT.getTypePtr();
        // if (TagDecl *TD = TP->getAsTagDecl()) {
        //   DLoc = TD->getLocation();
        // }
      } else if (TemplateSpecializationTypeLoc TSTL = TL->getAs<TemplateSpecializationTypeLoc>()) {
        if (auto *TST = TSTL.getTypePtr()->getAs<TemplateSpecializationType>()) {
          if (auto *TD = TST->getTemplateName().getAsTemplateDecl()) {
            DLoc = TD->getLocation();
          }
        }
      }
      PresumedLoc DPLoc = SM.getPresumedLoc(DLoc);
      if (DPLoc.isValid()) {
        Dfile = getPath(DPLoc.getFilename());
        Dline = DPLoc.getLine();
        Dcolumn = DPLoc.getColumn();
      }
#ifndef LOG
      if (exptype != "Elaborated" && !Dfile.empty() && skipPath(Dfile)) {
        stored_record_vec.clear();
        return;
      }/* else if (exptype == "TemplateSpecialization") {
        stored_record_vec.clear();
        return;
      }*/ else if (exptype == "Builtin") {
        stored_record_vec.clear();
        return;
      } else if (Loc.isInvalid() || SM.isInSystemHeader(Loc)) {
        stored_record_vec.clear();
        return;
      } else if (!file.empty() && skipPath(file)) {
        stored_record_vec.clear();
        return;
      }
#endif
      std::stringstream ss;
      ss << "{\"type\":\"TypeLoc\",\"file\":\"" << file
         << "\",\"line\":" << line << ",\"column\":" << column << ",\"stmt\":\""
         << stmt << "\",\"exptype\":\"" << exptype << "\",\"dfile\":\"" << Dfile
         << "\",\"dline\":" << Dline << ",\"dcolumn\":" << Dcolumn
         << ",\"dstmt\":\"" << Dstmt << "\"}";
      auto content = ss.str();
#ifndef LOG
      if (exptype == "Elaborated") {
        stored_record_vec.emplace_back(content);
      } else {
        for (const auto &it : stored_record_vec) {
          printf("%s\n", it.c_str());
        }
        stored_record_vec.clear();
        printf("%s\n", content.c_str());
      }
#else
      printf("%s\n", content.c_str());
#endif
      // if (exptype == "Enum")
      //   QT.dump();
    } else if (const DeclRefExpr *DRE =
                   Result.Nodes.getNodeAs<DeclRefExpr>("DeclRefExpr|")) {
      std::string file, Dfile, stmt, Dstmt;
      int line = 0, column = 0;
      int Dline = 0, Dcolumn = 0;
      SourceLocation Loc = SM.getSpellingLoc(DRE->getBeginLoc());
      if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
        return;
      {
        CharSourceRange FullRange =
            CharSourceRange::getCharRange(DRE->getBeginLoc(), DRE->getEndLoc());
        StringRef Text = Lexer::getSourceText(FullRange, SM, LangOptions());
        stmt = escapeJsonString(Text.str());
      }
      const ValueDecl *D = DRE->getDecl();
      SourceLocation DLoc = SM.getSpellingLoc(D->getBeginLoc());
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      if (PLoc.isValid()) {
        file = getPath(PLoc.getFilename());
        line = PLoc.getLine();
        column = PLoc.getColumn();
      }
      if (skipPath(file))
        return;
      PresumedLoc DPLoc = SM.getPresumedLoc(DLoc);
      if (DPLoc.isValid()) {
        Dfile = getPath(DPLoc.getFilename());
        Dline = DPLoc.getLine();
        Dcolumn = DPLoc.getColumn();
      }
      // if (!Dfile.empty() && skipPath(Dfile))
      //   return;
      {
        CharSourceRange FullRange =
            CharSourceRange::getCharRange(D->getBeginLoc(), D->getEndLoc());
        StringRef Text = Lexer::getSourceText(FullRange, SM, LangOptions());
        Dstmt = escapeJsonString(Text.str());
      }
      std::stringstream ss;
      ss << "{\"type\":\"DeclRefExpr\",\"file\":\"" << file
         << "\",\"line\":" << line << ",\"column\":" << column << ",\"stmt\":\""
         << stmt << "\",\"exptype\":\"" << DRE->getType()->getTypeClassName()
         << "\",\"dfile\":\"" << Dfile << "\",\"dline\":" << Dline
         << ",\"dcolumn\":" << Dcolumn << ",\"dstmt\":\"" << Dstmt << "\"}";
      printf("%s\n", ss.str().c_str());
    } else if (const CXXRecordDecl *CRD =
                   Result.Nodes.getNodeAs<CXXRecordDecl>("CXXRecordDecl|")) {
      std::string file, Dfile, stmt, Dstmt;
      int line = 0, column = 0;
      int Dline = 0, Dcolumn = 0;
      SourceLocation Loc = SM.getSpellingLoc(CRD->getLocation());
      if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
        return;
      stmt = escapeJsonString(CRD->getNameAsString());
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      if (PLoc.isValid()) {
        file = getPath(PLoc.getFilename());
        line = PLoc.getLine();
        column = PLoc.getColumn();
      }
      if (skipPath(file))
        return;
      std::stringstream ss;
      ss << "{\"type\":\"CXXRecordDecl\",\"file\":\"" << file
         << "\",\"line\":" << line << ",\"column\":" << column << ",\"stmt\":\""
         << stmt << "\",\"exptype\":\""
         << "CXXRecordDecl"
         << "\",\"dfile\":\"" << Dfile << "\",\"dline\":" << Dline
         << ",\"dcolumn\":" << Dcolumn << ",\"dstmt\":\"" << Dstmt << "\"}";
      printf("%s\n", ss.str().c_str());
    } else if (const FunctionDecl *FD =
                   Result.Nodes.getNodeAs<FunctionDecl>("FunctionDecl|")) {
      std::string file, Dfile, stmt, Dstmt;
      int line = 0, column = 0;
      int Dline = 0, Dcolumn = 0;
      SourceLocation Loc = SM.getSpellingLoc(FD->getBeginLoc());
      if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
        return;
      const CompoundStmt *Body = dyn_cast_or_null<CompoundStmt>(FD->getBody());
      if (Body != nullptr) {
        CharSourceRange FullRange = CharSourceRange::getCharRange(
            FD->getBeginLoc(), FD->getBody()->getBeginLoc());
        StringRef Text = Lexer::getSourceText(FullRange, SM, LangOptions());
        stmt = escapeJsonString(Text.str());
      } else {
        CharSourceRange FullRange =
            CharSourceRange::getCharRange(FD->getBeginLoc(), FD->getEndLoc());
        StringRef Text = Lexer::getSourceText(FullRange, SM, LangOptions());
        stmt = escapeJsonString(Text.str());
      }
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      if (PLoc.isValid()) {
        file = getPath(PLoc.getFilename());
        line = PLoc.getLine();
        column = PLoc.getColumn();
      }
      if (skipPath(file))
        return;
      SourceRange SR = FD->getReturnTypeSourceRange();
      StringRef Rstmt = Lexer::getSourceText(
          CharSourceRange::getTokenRange(SR.getBegin(), SR.getEnd()), SM,
          LangOptions());
      if (!stmt.empty() && !Rstmt.empty())
        stmt += "%%" + escapeJsonString(Rstmt.str());
      std::stringstream ss;
      ss << "{\"type\":\"FunctionDecl\",\"file\":\"" << file
         << "\",\"line\":" << line << ",\"column\":" << column << ",\"stmt\":\""
         << stmt << "\",\"exptype\":\""
         << (!FD->isThisDeclarationADefinition() || FD->getBody() == nullptr
                 ? "FunctionDecl"
                 : "FunctionImpl")
         << "\",\"dfile\":\"" << Dfile << "\",\"dline\":" << Dline
         << ",\"dcolumn\":" << Dcolumn << ",\"dstmt\":\"" << Dstmt << "\"}";
      printf("%s\n", ss.str().c_str());
    } else if (const auto *UDD = Result.Nodes.getNodeAs<UsingDirectiveDecl>(
                   "UsingDirectiveDecl|")) {
      std::string file, Dfile, stmt, Dstmt;
      int line = 0, column = 0;
      int Dline = 0, Dcolumn = 0;
      SourceLocation Loc = SM.getSpellingLoc(UDD->getBeginLoc());
      if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
        return;
      {
        SourceLocation End =
            Lexer::findLocationAfterToken(UDD->getEndLoc(), clang::tok::semi,
                                          SM, Context.getLangOpts(), false);
        if (End.isInvalid()) {
          End = Lexer::getLocForEndOfToken(UDD->getEndLoc(), 0, SM,
                                           Context.getLangOpts());
        }
        auto FullRange = CharSourceRange::getCharRange(UDD->getBeginLoc(), End);
        auto Text = Lexer::getSourceText(FullRange, SM, LangOptions());
        stmt = escapeJsonString(escapeJsonString(Text.str()));
      }
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      if (PLoc.isValid()) {
        file = getPath(PLoc.getFilename());
        line = PLoc.getLine();
        column = PLoc.getColumn();
      }
      if (skipPath(file))
        return;
      SourceLocation DLoc = UDD->getNominatedNamespace()->getBeginLoc();
      PresumedLoc DPLoc = SM.getPresumedLoc(DLoc);
      if (DPLoc.isValid()) {
        Dfile = getPath(DPLoc.getFilename());
        Dline = DPLoc.getLine();
        Dcolumn = DPLoc.getColumn();
      }
      if (!Dfile.empty() && skipPath(Dfile))
        return;
      std::stringstream ss;
      ss << "{\"type\":\"UsingDirectiveDecl\",\"file\":\"" << file
         << "\",\"line\":" << line << ",\"column\":" << column << ",\"stmt\":\""
         << stmt << "\",\"exptype\":\""
         << "UsingDirectiveDecl"
         << "\",\"dfile\":\"" << Dfile << "\",\"dline\":" << Dline
         << ",\"dcolumn\":" << Dcolumn << ",\"dstmt\":\"" << Dstmt << "\"}";
      printf("%s\n", ss.str().c_str());
    } else if (const NamedDecl *ND =
                   Result.Nodes.getNodeAs<NamedDecl>("NamedDecl|")) {
      std::string file, Dfile, stmt, Dstmt;
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
      if (skipPath(file))
        return;
      switch (ND->getKind()) {
      case NamedDecl::Var:
      case NamedDecl::Field:
      case NamedDecl::EnumConstant:
      case NamedDecl::ParmVar:
      case NamedDecl::Namespace:
        return;
      default:
        break;
      }
      stmt = escapeJsonString(ND->getNameAsString());
      std::stringstream ss;
      ss << "{\"type\":\"NamedDecl\",\"file\":\"" << file
         << "\",\"line\":" << line << ",\"column\":" << column << ",\"stmt\":\""
         << stmt << "\",\"exptype\":\""
         << "NamedDecl"
         << "\",\"dfile\":\"" << Dfile << "\",\"dline\":" << Dline
         << ",\"dcolumn\":" << Dcolumn << ",\"dstmt\":\"" << Dstmt << "\"}";
      printf("%s\n", ss.str().c_str());
    } else if (const CallExpr *CE =
                   Result.Nodes.getNodeAs<CallExpr>("CallExpr|")) {
      std::string file, Dfile, stmt, Dstmt;
      int line = 0, column = 0;
      int Dline = 0, Dcolumn = 0;
      SourceLocation Loc = SM.getSpellingLoc(CE->getBeginLoc());
      if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
        return;
      {
        CharSourceRange FullRange =
            CharSourceRange::getCharRange(CE->getBeginLoc(), CE->getEndLoc());
        StringRef Text = Lexer::getSourceText(FullRange, SM, LangOptions());
        stmt = escapeJsonString(Text.str());
      }
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      if (PLoc.isValid()) {
        file = getPath(PLoc.getFilename());
        line = PLoc.getLine();
        column = PLoc.getColumn();
      }
      if (skipPath(file))
        return;
      SourceLocation DLoc = Loc;
      if (const FunctionDecl *FD = CE->getDirectCallee()) {
        DLoc = FD->getBeginLoc();
        stmt += "%%" + FD->getNameAsString();
      }
      if (DLoc.isInvalid() || SM.isInSystemHeader(DLoc))
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
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef file) override {
    MatchFinder Finder;
    auto *Callback{
        new AnalysisMatchCallback{CI.getASTContext(), CI.getSourceManager()}};
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
    } else if (option.find("CXXRecordDecl|") != std::string::npos) {
      Finder.addMatcher(
          clang::ast_matchers::cxxRecordDecl().bind("CXXRecordDecl|"),
          Callback);
    } else if (option.find("FunctionDecl|") != std::string::npos) {
      Finder.addMatcher(
          clang::ast_matchers::functionDecl().bind("FunctionDecl|"), Callback);
    } else if (option.find("UsingDirectiveDecl|") != std::string::npos) {
      Finder.addMatcher(
          clang::ast_matchers::usingDirectiveDecl().bind("UsingDirectiveDecl|"),
          Callback);
    } else if (option.find("Stmt|") != std::string::npos) {
      Finder.addMatcher(clang::ast_matchers::stmt().bind("Stmt|"), Callback);
    } else if (option.find("NamedDecl|") != std::string::npos) {
      Finder.addMatcher(clang::ast_matchers::namedDecl().bind("NamedDecl|"),
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
