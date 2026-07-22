package jp.co.canon.bsd.ad.sdk.core.clss.struct;

import d7.InterfaceC1549a;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSConfigurationDeviceInfo {
    private static final String PREF_CLSSCR_PURPOSE_AGREEMENT_SERVICE = "_clsscr_purpose_agreement_service";
    private static final String PREF_CLSSPS_AUTO_PRINTER_ROM_UPDATE = "_clsscr_auto_printer_rom_update";
    private static final String PREF_CLSSPS_PLI_AGREEMENT = "_clsscr_pli_agreement";
    private static final String PREF_CLSSPS_PURPOSE_AGREEMENT_ANALYSIS = "_clsscr_purpose_agreement_analysis";
    private static final String PREF_CLSSPS_PURPOSE_AGREEMENT_VERSION = "_clsscr_purpose_agreement_version";
    private static final String PREF_CLSSPS_QUESTIONNAIRE_STATE = "_clsscr_questionnaire_state";
    private static final String PREF_CLSSPS_REDUCE_INK_SMUDGES = "_clsscr_reduce_ink_smudges";
    private static final String PREF_CLSSPS_WEBSERVICE_AGREEMENT = "_clsscr_webservice_agreement";

    @InterfaceC1549a(defInt = 65535, key = PREF_CLSSPS_AUTO_PRINTER_ROM_UPDATE)
    public int auto_printer_rom_update;

    @InterfaceC1549a(defInt = 65535, key = PREF_CLSSPS_PLI_AGREEMENT)
    public int pli_agreement;

    @InterfaceC1549a(defInt = 65535, key = PREF_CLSSPS_PURPOSE_AGREEMENT_ANALYSIS)
    public int purpose_agreement_analysis;

    @InterfaceC1549a(defInt = 65535, key = PREF_CLSSCR_PURPOSE_AGREEMENT_SERVICE)
    public int purpose_agreement_service;

    @InterfaceC1549a(defInt = 65535, key = PREF_CLSSPS_PURPOSE_AGREEMENT_VERSION)
    public int purpose_agreement_version;

    @InterfaceC1549a(defInt = 65535, key = PREF_CLSSPS_QUESTIONNAIRE_STATE)
    public int questionnaire_state;

    @InterfaceC1549a(defInt = 65535, key = PREF_CLSSPS_REDUCE_INK_SMUDGES)
    public int reduce_ink_smudges;

    @InterfaceC1549a(defInt = 65535, key = PREF_CLSSPS_WEBSERVICE_AGREEMENT)
    public int webservice_agreement;

    public CLSSConfigurationDeviceInfo() {
        init();
    }

    public int getAutoPrinterRomUpdate() {
        return this.auto_printer_rom_update;
    }

    public int getPli_agreement() {
        return this.pli_agreement;
    }

    public int getPurposeAgreementAnalysis() {
        return this.purpose_agreement_analysis;
    }

    public int getPurposeAgreementService() {
        return this.purpose_agreement_service;
    }

    public int getPurposeAgreementVersion() {
        return this.purpose_agreement_version;
    }

    public int getQuestionnaire_state() {
        return this.questionnaire_state;
    }

    public int getWebservice_agreement() {
        return this.webservice_agreement;
    }

    public void init() {
        set(65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535);
    }

    public void set(int i9, int i10, int i11, int i12, int i13, int i14, int i15, int i16) {
        this.pli_agreement = i9;
        this.webservice_agreement = i10;
        this.questionnaire_state = i11;
        this.auto_printer_rom_update = i12;
        this.purpose_agreement_service = i13;
        this.purpose_agreement_analysis = i14;
        this.purpose_agreement_version = i15;
        this.reduce_ink_smudges = i16;
    }

    public void setAutoPrinterRomUpdate(int i9) {
        this.auto_printer_rom_update = i9;
    }

    public void setPli_agreement(int i9) {
        this.pli_agreement = i9;
    }

    public void setPurposeAgreementAnalysis(int i9) {
        this.purpose_agreement_analysis = i9;
    }

    public void setPurposeAgreementService(int i9) {
        this.purpose_agreement_service = i9;
    }

    public void setPurposeAgreementVersion(int i9) {
        this.purpose_agreement_version = i9;
    }

    public void setQuestionnaire_state(int i9) {
        this.questionnaire_state = i9;
    }

    public void setReduceInkSmudges(int i9) {
        this.reduce_ink_smudges = i9;
    }

    public void setWebservice_agreement(int i9) {
        this.webservice_agreement = i9;
    }

    public CLSSConfigurationDeviceInfo(int i9, int i10, int i11, int i12, int i13, int i14, int i15, int i16) {
        set(i9, i10, i11, i12, i13, i14, i15, i16);
    }
}
