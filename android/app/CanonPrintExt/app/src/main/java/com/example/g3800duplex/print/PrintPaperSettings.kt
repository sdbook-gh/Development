package com.example.g3800duplex.print

import android.content.Context
import android.print.PrintAttributes
import jp.co.canon.bsd.ad.sdk.core.clss.CLSS_Define

enum class PaperSize(
    val label: String,
    /** Width / height in millimetres (portrait). */
    val widthMm: Float,
    val heightMm: Float,
    val clssSize: Int,
    val ippMedia: String,
    val cloudMediaSize: String,
) {
    A4(
        "A4 (210×297mm)",
        210f, 297f,
        CLSS_Define.CLSS_IVEC_SIZE_A4,
        "iso_a4_210x297mm",
        "cpk2:iso_a4_210x297mm",
    ),
    A5(
        "A5 (148×210mm)",
        148f, 210f,
        CLSS_Define.CLSS_IVEC_SIZE_A5,
        "iso_a5_148x210mm",
        "cpk2:iso_a5_148x210mm",
    ),
    B5(
        "B5 (176×250mm)",
        176f, 250f,
        CLSS_Define.CLSS_IVEC_SIZE_B5,
        "iso_b5_176x250mm",
        "cpk2:jis_b5_182x257mm",
    ),
    Letter(
        "Letter (8.5×11in)",
        215.9f, 279.4f,
        CLSS_Define.CLSS_IVEC_SIZE_LETTER,
        "na_letter_8.5x11in",
        "cpk2:na_letter_8.5x11in",
    ),
    Legal(
        "Legal (8.5×14in)",
        215.9f, 355.6f,
        CLSS_Define.CLSS_IVEC_SIZE_LEGAL,
        "na_legal_8.5x14in",
        "cpk2:na_legal_8.5x14in",
    ),
    L(
        "L (89×127mm)",
        89f, 127f,
        CLSS_Define.CLSS_IVEC_SIZE_L,
        "jpn_hagaki_100x148mm",
        "cpk2:jpn_l_89x127mm",
    ),
    TwoL(
        "2L (127×178mm)",
        127f, 178f,
        CLSS_Define.CLSS_IVEC_SIZE_2L,
        "oe_photo-l_3.5x5in",
        "cpk2:jpn_2l_127x178mm",
    ),
    Photo4x6(
        "4×6in (10×15cm)",
        101.6f, 152.4f,
        CLSS_Define.CLSS_IVEC_SIZE_4X6,
        "na_index-4x6_4x6in",
        "cpk2:na_index_4x6in",
    ),
    Photo5x7(
        "5×7in (13×18cm)",
        127f, 177.8f,
        CLSS_Define.CLSS_IVEC_SIZE_5X7,
        "na_5x7_5x7in",
        "cpk2:na_5x7_5x7in",
    ),
    ;

    /** PDF page size at 72 dpi. */
    val pdfWidthPt: Int get() = mmToPt(widthMm)
    val pdfHeightPt: Int get() = mmToPt(heightMm)

    fun androidMediaSize(): PrintAttributes.MediaSize = when (this) {
        A4 -> PrintAttributes.MediaSize.ISO_A4
        A5 -> PrintAttributes.MediaSize.ISO_A5
        B5 -> PrintAttributes.MediaSize.JIS_B5
        Letter -> PrintAttributes.MediaSize.NA_LETTER
        Legal -> PrintAttributes.MediaSize.NA_LEGAL
        Photo4x6 -> PrintAttributes.MediaSize.NA_INDEX_4X6
        Photo5x7 -> customMedia("na_5x7_5x7in", "5x7", 5000, 7000)
        L -> customMedia("jpn_l_89x127mm", "L", 3504, 5000)
        TwoL -> customMedia("jpn_2l_127x178mm", "2L", 5000, 7008)
    }

    private fun customMedia(
        id: String,
        label: String,
        widthMils: Int,
        heightMils: Int,
    ): PrintAttributes.MediaSize =
        PrintAttributes.MediaSize(id, label, widthMils, heightMils)

    companion object {
        fun mmToPt(mm: Float): Int = (mm * 72f / 25.4f).toInt().coerceAtLeast(1)

        fun fromName(name: String?): PaperSize =
            entries.firstOrNull { it.name == name } ?: A4
    }
}

enum class PaperMedia(
    val label: String,
    val clssMedia: Int,
    /** Cloud PrintTicket MediaTypeClass option. */
    val cloudMediaClass: String,
) {
    Plain("普通纸", CLSS_Define.CLSS_IVEC_MEDIA_PLAIN, "cpk2:stationary"),
    Glossy("光面照片纸", CLSS_Define.CLSS_IVEC_MEDIA_GLOSSY_PAPER, "cpk2:photo"),
    Matte("哑光纸", CLSS_Define.CLSS_IVEC_MEDIA_MATTE_PAPER, "cpk2:photo"),
    Photo("照片纸", CLSS_Define.CLSS_IVEC_MEDIA_PHOTOPAPER, "cpk2:photo"),
    Envelope("信封", CLSS_Define.CLSS_IVEC_MEDIA_ENVELOPE, "cpk2:stationary"),
    ;

    companion object {
        fun fromName(name: String?): PaperMedia =
            entries.firstOrNull { it.name == name } ?: Plain
    }
}

data class PrintPaperSettings(
    val size: PaperSize = PaperSize.A4,
    val media: PaperMedia = PaperMedia.Plain,
) {
    fun summary(): String = "${size.label} · ${media.label}"
}

class PaperSettingsStore(context: Context) {
    private val prefs = context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    fun load(): PrintPaperSettings {
        val size = PaperSize.fromName(prefs.getString(KEY_SIZE, PaperSize.A4.name))
        val media = PaperMedia.fromName(prefs.getString(KEY_MEDIA, PaperMedia.Plain.name))
        return PrintPaperSettings(size, media)
    }

    fun save(settings: PrintPaperSettings) {
        prefs.edit()
            .putString(KEY_SIZE, settings.size.name)
            .putString(KEY_MEDIA, settings.media.name)
            .apply()
    }

    companion object {
        private const val PREFS = "g3800_paper_settings"
        private const val KEY_SIZE = "paper_size"
        private const val KEY_MEDIA = "paper_media"
    }
}
