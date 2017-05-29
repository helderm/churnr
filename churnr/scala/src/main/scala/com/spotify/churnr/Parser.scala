package com.spotify.churnr

/**
  * Created by helderm on 2017-05-18.
  */

import com.spotify.scio.bigquery._
import com.spotify.scio._
import org.slf4j.{Logger, LoggerFactory}
import com.spotify.jamalytics.play_context.{ContentType, PlayContextParser, TopType}
import com.spotify.scio.values.SCollection

import scala.annotation.StaticAnnotation

/*
sbt -Dbigquery.project=[PROJECT]
runMain
  com.spotify.scio.examples.extra.TypedBigQueryTornadoes
  --project=[PROJECT] --runner=DataflowRunner --zone=[ZONE]
  --output=[DATASET].typed_bigquery_tornadoes
*/

object Parser {

  @BigQueryType.fromTable("helderm.features_coolexp_7_7d_s")
  class Coolexp_7_7d

  @BigQueryType.fromTable("helderm.features_coolexp_14_7d_s")
  class Coolexp_14_7d

  @BigQueryType.fromTable("helderm.features_temporal_static_60_30d_s")
  class TemporalStatic_60_30d

  @BigQueryType.toTable
  case class Result(user_id: String, timestampp: Long, skipped:Boolean, secs_played: Long, play_context_decrypted: String,
                    top_type: String, sub_type: String, platform: String, latitude: Double, longitude: Double, client_type: String)

  def main(cmdlineArgs: Array[String]): Unit = {
    val (sc, args) = ContextAndArgs(cmdlineArgs)

    val log: Logger = LoggerFactory.getLogger(Parser.getClass)

    log.info("Starting extraction of play context data...")

    if (args("output").endsWith("coolexp_7_7d_p")) {
      sc.typedBigQuery[Coolexp_7_7d]()
        .map(r => (r.user_id, r.timestampp, r.skipped, r.secs_played, r.play_context_decrypted,
                    PlayContextParser.parse(r.play_context_decrypted.getOrElse(null)),
                    r.platform, r.latitude, r.longitude, r.client_type))
        .map(kv => Result(kv._1.getOrElse(null), kv._2.getOrElse(0L), kv._3.getOrElse(false), kv._4.getOrElse(0L),
          kv._5.getOrElse(null), kv._6.getOrElse(new ContentType("", TopType.Unknown)).topType.toString,
          kv._6.getOrElse(new ContentType("", TopType.Unknown)).subType.toString, kv._7.getOrElse(null),
          kv._8.getOrElse(999.9), kv._9.getOrElse(999.9), kv._10.getOrElse(null)))
        .saveAsTypedBigQuery(args("output"), WRITE_TRUNCATE, CREATE_IF_NEEDED)
    } else if (args("output").endsWith("coolexp_14_7d_p")) {
      sc.typedBigQuery[Coolexp_14_7d]()
        .map(r => (r.user_id, r.timestampp, r.skipped, r.secs_played, r.play_context_decrypted,
          PlayContextParser.parse(r.play_context_decrypted.getOrElse(null)),
          r.platform, r.latitude, r.longitude, r.client_type))
        .map(kv => Result(kv._1.getOrElse(null), kv._2.getOrElse(0L), kv._3.getOrElse(false), kv._4.getOrElse(0L),
          kv._5.getOrElse(null), kv._6.getOrElse(new ContentType("", TopType.Unknown)).topType.toString,
          kv._6.getOrElse(new ContentType("", TopType.Unknown)).subType.toString, kv._7.getOrElse(null),
          kv._8.getOrElse(999.9), kv._9.getOrElse(999.9), kv._10.getOrElse(null)))
        .saveAsTypedBigQuery(args("output"), WRITE_TRUNCATE, CREATE_IF_NEEDED)
    } else if (args("output").endsWith("temporal_static_60_30d_p")) {
      sc.typedBigQuery[TemporalStatic_60_30d]()
        .map(r => (r.user_id, r.timestampp, r.skipped, r.secs_played, r.play_context_decrypted,
          PlayContextParser.parse(r.play_context_decrypted.getOrElse(null)),
          r.platform, r.latitude, r.longitude, r.client_type))
        .map(kv => Result(kv._1.getOrElse(null), kv._2.getOrElse(0L), kv._3.getOrElse(false), kv._4.getOrElse(0L),
          kv._5.getOrElse(null), kv._6.getOrElse(new ContentType("", TopType.Unknown)).topType.toString,
          kv._6.getOrElse(new ContentType("", TopType.Unknown)).subType.toString, kv._7.getOrElse(null),
          kv._8.getOrElse(999.9), kv._9.getOrElse(999.9), kv._10.getOrElse(null)))
        .saveAsTypedBigQuery(args("output"), WRITE_TRUNCATE, CREATE_IF_NEEDED)
    } else {
      throw new Exception("Invalid output!")
    }

    val result = sc.close()

    log.info("Waiting for Dataflow job to finish...")
    result.waitUntilDone()
    log.info("Play context parsed successfully!")
  }

}
