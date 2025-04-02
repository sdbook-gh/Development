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
#include <sstream>
#include <string>
#include <vector>

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
  if (realPath.find("/binaries/") != std::string::npos) {
    return true;
  }
  if (realPath.find("/build/") != std::string::npos) {
    return true;
  }
  if (realPath.find("/Qt/") != std::string::npos) {
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
    }
    ss << "{\"type\":\"MacroDefined\",\"file\":\"" << file
       << "\",\"line\":" << line << ",\"column\":" << column
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

  void Ifndef(SourceLocation Loc, const Token &MacroNameTok,
              const MacroDefinition &MD) override {
    if (option.find("MacroDefExp|") == std::string::npos)
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

  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange,
                          OptionalFileEntryRef File, StringRef SearchPath,
                          StringRef RelativePath, const Module *Imported,
                          SrcMgr::CharacteristicKind FileType) override {
    if (option.find("MacroDefExp|") == std::string::npos)
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
      if (!matchFileContent(file, line, column, raw_stmt)) {
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
      if (Loc.isInvalid() || SM.isInSystemHeader(Loc))
        return;
      std::string exptype = DRE->getType()->getTypeClassName();
      {
        CharSourceRange FullRange;
        if (DRE->getBeginLoc().isMacroID()) {
          FullRange = CharSourceRange::getTokenRange(
              SM.getSpellingLoc(DRE->getBeginLoc()),
              SM.getSpellingLoc(DRE->getEndLoc()));
        } else {
          FullRange = CharSourceRange::getTokenRange(DRE->getBeginLoc(),
                                                     DRE->getEndLoc());
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
      Dstmt = escapeJsonString("DeclRefExpr");
      std::stringstream ss;
      if (skipPath(file)) {
        ss << "##";
      } else if (!Dfile.empty() && skipPath(Dfile)) {
        ss << "##";
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
      Dstmt = escapeJsonString("TypeLoc");
      QualType QT = TL->getType();
      std::string exptype = QT->getTypeClassName();
      if (skipPath(file)) {
        ss << "####";
      } else if (!Dfile.empty() && skipPath(Dfile)) {
        ss << "####";
      } else if (!matchFileContent(file, line, column, raw_stmt)) {
        ss << "####";
      }
      ss << "{\"type\":\"DeclRefExprTypeLoc\",\"file\":\"" << file
         << "\",\"line\":" << line << ",\"column\":" << column << ",\"stmt\":\""
         << stmt << "\",\"exptype\":\"" << exptype << "\",\"dfile\":\"" << Dfile
         << "\",\"dline\":" << Dline << ",\"dcolumn\":" << Dcolumn
         << ",\"dstmt\":\"" << Dstmt << "\"}";
      auto content = ss.str();
      printf("%s\n", content.c_str());
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
        CharSourceRange FullRange = CharSourceRange::getCharRange(
            FD->getBeginLoc(), FD->getBody()->getBeginLoc());
        raw_stmt = Lexer::getSourceText(FullRange, SM, LangOptions()).str();
        stmt = escapeJsonString(raw_stmt);
      } else {
        CharSourceRange FullRange =
            CharSourceRange::getCharRange(FD->getBeginLoc(), FD->getEndLoc());
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
        ss << "##";
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
        SourceLocation End =
            Lexer::findLocationAfterToken(UDD->getEndLoc(), clang::tok::semi,
                                          SM, Context.getLangOpts(), false);
        if (End.isInvalid()) {
          End = Lexer::getLocForEndOfToken(UDD->getEndLoc(), 0, SM,
                                           Context.getLangOpts());
        }
        auto FullRange = CharSourceRange::getCharRange(UDD->getBeginLoc(), End);
        raw_stmt = Lexer::getSourceText(FullRange, SM, LangOptions()).str();
        stmt = escapeJsonString(escapeJsonString(raw_stmt));
      }
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      if (PLoc.isValid()) {
        file = getPath(PLoc.getFilename());
        line = PLoc.getLine();
        column = PLoc.getColumn();
      }
      // if (skipPath(file))
      //   return;
      SourceLocation DLoc = UDD->getNominatedNamespace()->getBeginLoc();
      if (DLoc.isInvalid() || SM.isInSystemHeader(DLoc))
        return;
      PresumedLoc DPLoc = SM.getPresumedLoc(DLoc);
      if (DPLoc.isValid()) {
        Dfile = getPath(DPLoc.getFilename());
        Dline = DPLoc.getLine();
        Dcolumn = DPLoc.getColumn();
      }
      // if (!Dfile.empty() && skipPath(Dfile))
      //   return;
      std::stringstream ss;
      if (!matchFileContent(file, line, column, raw_stmt)) {
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
      Dstmt = ND->getDeclKindName();
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
        ss << "##";
      } else if (!matchFileContent(file, line, column, raw_stmt)) {
        ss << "####";
      }
      ss << "{\"type\":\"NamedDeclMemberExpr\",\"file\":\"" << file
         << "\",\"line\":" << line << ",\"column\":" << column << ",\"stmt\":\""
         << stmt << "\",\"exptype\":\""
         << "NamedDecl"
         << "\",\"dfile\":\"" << Dfile << "\",\"dline\":" << Dline
         << ",\"dcolumn\":" << Dcolumn << ",\"dstmt\":\"" << Dstmt << "\"}";
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
            CharSourceRange::getCharRange(CE->getBeginLoc(), CE->getEndLoc());
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
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef file) override {
    MatchFinder Finder;
    auto *Callback{
        new AnalysisMatchCallback{CI.getASTContext(), CI.getSourceManager()}};
    if (option.find("NamespaceDecl|") != std::string::npos) {
      Finder.addMatcher(
          clang::ast_matchers::namespaceDecl().bind("NamespaceDecl|"),
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
    } else if (option.find("Stmt|") != std::string::npos) {
      Finder.addMatcher(clang::ast_matchers::stmt().bind("Stmt|"), Callback);
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
