@rem
@rem Gradle wrapper script for Windows
@rem

@if "%DEBUG%"=="" @echo off
@rem ##########################################################################
@rem  Gradle start up script for Windows
@rem ##########################################################################

setlocal

set APP_HOME=%~dp0

set CLASSPATH=%APP_HOME%\gradle\wrapper\gradle-wrapper.jar

set DEFAULT_JVM_OPTS="-Xmx64m" "-Xms64m"

set JAVA_EXE=java.exe

"%JAVA_EXE%" %DEFAULT_JVM_OPTS% "-Dorg.gradle.appname=%~n0" -classpath "%CLASSPATH%" org.gradle.wrapper.GradleWrapperMain %*