name := "churnr"

version := "1.0"

scalaVersion := "2.11.1"

libraryDependencies ++= Seq(
  "com.spotify" % "scio-core_2.11" % "0.3.0",
  "com.spotify" % "scio-test_2.11" % "0.3.0" % "test",
  "com.spotify.jamalytics" % "play-context-parser_2.11" % "0.8.2"
)

addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)