package jp.co.canon.bsd.ad.sdk.core.clss.struct;


/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSSetConfigurationWithoutStartJobParam {
    private String mPrinterSetupCurrentSupportCode;
    private String mPrinterSetupMainStatus;
    private String mPrinterSetupSupportParam;
    private int mPurposeAgreementAnalysis;
    private int mPurposeAgreementService;
    private int mPurposeAgreementVersion;
    private int mWebserviceAgreement;

    public CLSSSetConfigurationWithoutStartJobParam() {
        this.mPrinterSetupMainStatus = null;
        this.mPrinterSetupSupportParam = null;
        this.mPrinterSetupCurrentSupportCode = null;
        this.mPurposeAgreementService = 65535;
        this.mPurposeAgreementAnalysis = 65535;
        this.mPurposeAgreementVersion = 65535;
        this.mWebserviceAgreement = 65535;
    }

    public String getPrinterSetupCurrentSupportCode() {
        return this.mPrinterSetupCurrentSupportCode;
    }

    public String getPrinterSetupMainStatus() {
        return this.mPrinterSetupMainStatus;
    }

    public String getPrinterSetupSupportParam() {
        return this.mPrinterSetupSupportParam;
    }

    public int getPurposeAgreementAnalysis() {
        return this.mPurposeAgreementAnalysis;
    }

    public int getPurposeAgreementService() {
        return this.mPurposeAgreementService;
    }

    public int getPurposeAgreementVersion() {
        return this.mPurposeAgreementVersion;
    }

    public int getWebserviceAgreement() {
        return this.mWebserviceAgreement;
    }

    public void setPrinterSetupCurrentSupportCode(String str) {
        this.mPrinterSetupCurrentSupportCode = str;
    }

    public void setPrinterSetupMainStatus(String str) {
        this.mPrinterSetupMainStatus = str;
    }

    public void setPrinterSetupSupportParam(String str) {
        this.mPrinterSetupSupportParam = str;
    }

    public void setPurposeAgreementAnalysis(int i9) {
        this.mPurposeAgreementAnalysis = i9;
    }

    public void setPurposeAgreementService(int i9) {
        this.mPurposeAgreementService = i9;
    }

    public void setPurposeAgreementVersion(int i9) {
        this.mPurposeAgreementVersion = i9;
    }

    public void setWebserviceAgreement(int i9) {
        this.mWebserviceAgreement = i9;
    }

    public CLSSSetConfigurationWithoutStartJobParam(String str, String str2, String str3, int i9, int i10, int i11, int i12) {
        this.mPrinterSetupMainStatus = str;
        this.mPrinterSetupSupportParam = str2;
        this.mPrinterSetupCurrentSupportCode = str3;
        this.mPurposeAgreementService = i9;
        this.mPurposeAgreementAnalysis = i10;
        this.mPurposeAgreementVersion = i11;
        this.mWebserviceAgreement = i12;
    }

    public CLSSSetConfigurationWithoutStartJobParam(int i9, int i10, int i11) {
        this.mPrinterSetupMainStatus = null;
        this.mPrinterSetupSupportParam = null;
        this.mPrinterSetupCurrentSupportCode = null;
        this.mPurposeAgreementService = i9;
        this.mPurposeAgreementAnalysis = i10;
        this.mPurposeAgreementVersion = i11;
        this.mWebserviceAgreement = 65535;
    }

    public CLSSSetConfigurationWithoutStartJobParam(String str, String str2, String str3) {
        this.mPrinterSetupMainStatus = str;
        this.mPrinterSetupSupportParam = str2;
        this.mPrinterSetupCurrentSupportCode = str3;
        this.mPurposeAgreementService = 65535;
        this.mPurposeAgreementAnalysis = 65535;
        this.mPurposeAgreementVersion = 65535;
        this.mWebserviceAgreement = 65535;
    }

    public CLSSSetConfigurationWithoutStartJobParam(String str, String str2) {
        this.mPrinterSetupMainStatus = str;
        this.mPrinterSetupSupportParam = str2;
        this.mPrinterSetupCurrentSupportCode = null;
        this.mPurposeAgreementService = 65535;
        this.mPurposeAgreementAnalysis = 65535;
        this.mPurposeAgreementVersion = 65535;
        this.mWebserviceAgreement = 65535;
    }
}
