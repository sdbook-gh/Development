package jp.co.canon.bsd.ad.sdk.core.clss.struct;

import android.text.TextUtils;
import android.util.Base64;
import d7.InterfaceC1549a;
import jp.co.canon.bsd.ad.sdk.core.util.struct.CipherData;
import k7.C1791a;

/* loaded from: /mnt/f/print/classes3.dex */
public class CLSSCapabilityDeviceInfo {
    private static final String PREF_DI_AUTO_PRINTER_ROM_UPDATE = "_clsscrd_auto_printer_rom_update";
    private static final String PREF_DI_BOX_NUM = "_clsscrd_box_num";
    private static final String PREF_DI_CPC_VERSION = "_clsscrd_cpc_version";
    private static final String PREF_DI_FLG_BIN_INFO_SET_TABLE = "_clsscrd_flg_bin_info_set_table";
    private static final String PREF_DI_FLG_BLEMODE_AUTOLAUNCH_DIRECTMODE = "_clsscrd_flg_blemode_autolaunch_directmode";
    private static final String PREF_DI_FLG_COMMUNICATION_BAR = "_clsscrd_flg_communication_bar";
    private static final String PREF_DI_FLG_HDD = "_clsscrd_flg_hdd";
    private static final String PREF_DI_FLG_NETWORK_SETTINGS = "_clsscrd_flg_network_settings";
    private static final String PREF_DI_FLG_PASSWORD_SKIP_WIFI = "_clsscrd_flg_password_skip_wifi";
    private static final String PREF_DI_FLG_PRINTER_SET_UP = "_clsscrd_flg_printer_set_up";
    private static final String PREF_DI_FLG_REDUCE_INK_SMUDGES = "_clsscrd_flg_reduce_ink_smudges";
    private static final String PREF_DI_FLG_REMOTE_BACK_TO_APP = "_clsscrd_flg_remote_back_to_app";
    private static final String PREF_DI_FLG_REMOTE_CERTIFICATION_GUIDE = "_clsscrd_flg_remote_certification_guide";
    private static final String PREF_DI_FLG_REMOTE_UI = "_clsscrd_flg_remote_ui";
    private static final String PREF_DI_FLG_SET_CONFIGURATION_WITHOUT_START_JOB = "_clsscrd_flg_set_configuration_without_start_job";
    private static final String PREF_DI_FLG_SHOWABLE_WEBVIEW = "_clsscrd_flg_showable_webview";
    private static final String PREF_DI_FLG_SUBSCRIPTION_STATUS = "_clsscrd_flg_printer_subscription_status";
    private static final String PREF_DI_FLG_WEB_MANUAL = "_clsscrd_flg_web_manual";
    private static final String PREF_DI_FLG_WIRELESS_CONNECT_BUTTON = "_clsscrd_flg_wireless_connect_button";
    private static final String PREF_DI_HRI_ID = "_clsscrd_hri_id";
    private static final String PREF_DI_MQTT_CONNECTION = "_clsscrd_mqtt_connection";
    private static final String PREF_DI_PASSWORD_SKIP_WIFI_TYPE = "_clsscrd_password_skip_wifi_type";
    private static final String PREF_DI_PDR_ID = "_clsscrd_pdr_id";
    private static final String PREF_DI_PLI_AGREEMENT_ID = "_clsscrd_pli_agreement_id";
    private static final String PREF_DI_PRODUCT_SERIALNUMBER = "_clsscrd_product_serialnumber";
    private static final String PREF_DI_PRODUCT_SERIALNUMBER_ANGO = "ahoujhdshfodhfoiewjfoiejwfioejfoiwe";
    private static final String PREF_DI_PRODUCT_SERIALNUMBER_ANGO_KAGI = "yiuiuhudfihmpdmvpsojkpodsjfipdsj";
    private static final String PREF_DI_QUESTIONNAIRE_STATE = "_clsscrd_questionnaire_state";
    private static final String PREF_DI_REMOTEUI_LINK_TYPEB = "_clsscrd_remoteui_link_typeB";
    private static final String PREF_DI_REMOTE_UI_LINK = "_clsscrd_remote_ui_link";
    private static final String PREF_DI_REMOTE_UI_LINK_TYPEA = "_clsscrd_remote_ui_link_typea";
    private static final String PREF_DI_SUB_MODEL = "_clsscrd_sub_model";
    private static final String PREF_DI_WEBSERVICE_AGREEMENT = "_clsscrd_webservice_agreement";

    @InterfaceC1549a(key = PREF_DI_AUTO_PRINTER_ROM_UPDATE)
    public int[] auto_printer_rom_update;

    @InterfaceC1549a(key = PREF_DI_BOX_NUM)
    public int box_num;

    @InterfaceC1549a(key = PREF_DI_CPC_VERSION)
    public String cpc_version;

    @InterfaceC1549a(key = PREF_DI_PRODUCT_SERIALNUMBER_ANGO)
    private String encrypted_product_serialnumber;

    @InterfaceC1549a(key = PREF_DI_PRODUCT_SERIALNUMBER_ANGO_KAGI)
    private String encrypted_product_serialnumber_key;

    @InterfaceC1549a(key = PREF_DI_FLG_BIN_INFO_SET_TABLE)
    public boolean flg_bininfo_settable;

    @InterfaceC1549a(key = PREF_DI_FLG_BLEMODE_AUTOLAUNCH_DIRECTMODE)
    public boolean flg_blemode_autolaunch_directmode;

    @InterfaceC1549a(key = PREF_DI_FLG_COMMUNICATION_BAR)
    public boolean flg_communication_bar;

    @InterfaceC1549a(key = PREF_DI_FLG_HDD)
    public boolean flg_hdd;

    @InterfaceC1549a(key = PREF_DI_FLG_NETWORK_SETTINGS)
    public boolean flg_network_settings;

    @InterfaceC1549a(key = PREF_DI_FLG_PASSWORD_SKIP_WIFI)
    public boolean flg_password_skip_wifi;

    @InterfaceC1549a(key = PREF_DI_FLG_PRINTER_SET_UP)
    public boolean flg_printer_set_up;

    @InterfaceC1549a(key = PREF_DI_FLG_SUBSCRIPTION_STATUS)
    public boolean flg_printer_subscription_status;

    @InterfaceC1549a(key = PREF_DI_FLG_REDUCE_INK_SMUDGES)
    public boolean flg_reduce_ink_smudges;

    @InterfaceC1549a(key = PREF_DI_FLG_REMOTE_BACK_TO_APP)
    public boolean flg_remote_backtoapp;

    @InterfaceC1549a(key = PREF_DI_FLG_REMOTE_CERTIFICATION_GUIDE)
    public boolean flg_remote_certificationguide;

    @InterfaceC1549a(key = PREF_DI_FLG_REMOTE_UI)
    public boolean flg_remote_ui;

    @InterfaceC1549a(key = PREF_DI_FLG_SET_CONFIGURATION_WITHOUT_START_JOB)
    public boolean flg_set_configuration_without_start_job;

    @InterfaceC1549a(key = PREF_DI_FLG_SHOWABLE_WEBVIEW)
    public boolean flg_showable_webview;

    @InterfaceC1549a(key = PREF_DI_FLG_WEB_MANUAL)
    public boolean flg_web_manual;

    @InterfaceC1549a(key = PREF_DI_FLG_WIRELESS_CONNECT_BUTTON)
    public boolean flg_wireless_connect_button;

    @InterfaceC1549a(key = PREF_DI_HRI_ID)
    public String hriID;

    @InterfaceC1549a(key = PREF_DI_MQTT_CONNECTION)
    public int[] mqtt_connection;

    @InterfaceC1549a(key = PREF_DI_PASSWORD_SKIP_WIFI_TYPE)
    public int password_skip_wifi_type;

    @InterfaceC1549a(key = PREF_DI_PDR_ID)
    public String pdrID;

    @InterfaceC1549a(key = PREF_DI_PLI_AGREEMENT_ID)
    public int[] pli_agreementID;

    @InterfaceC1549a(key = PREF_DI_PRODUCT_SERIALNUMBER)
    private String product_serialnumber;

    @InterfaceC1549a(key = PREF_DI_QUESTIONNAIRE_STATE)
    public int[] questionnaire_state;

    @InterfaceC1549a(key = PREF_DI_REMOTE_UI_LINK)
    public String[] remoteui_link;

    @InterfaceC1549a(key = PREF_DI_REMOTE_UI_LINK_TYPEA)
    public String[] remoteui_link_typea;

    @InterfaceC1549a(key = PREF_DI_REMOTEUI_LINK_TYPEB)
    public String[] remoteui_link_typeb;

    @InterfaceC1549a(key = PREF_DI_SUB_MODEL)
    public int sub_model;

    @InterfaceC1549a(key = PREF_DI_WEBSERVICE_AGREEMENT)
    public int[] webservice_agreement;

    public CLSSCapabilityDeviceInfo() {
        init();
    }

        private String decryptProductSerialnumber() {
        if (TextUtils.isEmpty(this.encrypted_product_serialnumber)) {
            return "";
        }
        try {
            return C1791a.a(new CipherData(Base64.decode(this.encrypted_product_serialnumber.getBytes(), 0), Base64.decode(this.encrypted_product_serialnumber_key.getBytes(), 0)), PREF_DI_PRODUCT_SERIALNUMBER, PREF_DI_PLI_AGREEMENT_ID);
        } catch (Exception unused) {
            return "";
        }
    }

    private String newString(String str) {
        if (str == null) {
            return null;
        }
        try {
            return new String(str);
        } catch (Exception unused) {
            return null;
        }
    }

    public void encryptProductSerialnumberIfNeeded() {
        if (TextUtils.isEmpty(this.product_serialnumber)) {
            return;
        }
        try {
            CipherData b10 = C1791a.b(this.product_serialnumber, PREF_DI_PRODUCT_SERIALNUMBER, PREF_DI_PLI_AGREEMENT_ID);
            this.encrypted_product_serialnumber = Base64.encodeToString(b10.getData(), 0);
            this.encrypted_product_serialnumber_key = Base64.encodeToString(b10.getParams(), 0);
            this.product_serialnumber = "";
        } catch (Exception e10) {
            e10.toString();
        }
    }

    public String getProductSerialnumber() {
        encryptProductSerialnumberIfNeeded();
        return decryptProductSerialnumber();
    }

    public void init() {
        set(null, false, null, null, false, false, false, 65535, null, null, false, false, null, null, false, 65535, null, false, false, false, false, false, null, 65535, null, false, null, false, null, false, false);
    }

    public void set(int[] iArr, boolean z9, String str, String str2, boolean z10, boolean z11, boolean z12, int i9, String[] strArr, String[] strArr2, boolean z13, boolean z14, int[] iArr2, int[] iArr3, boolean z15, int i10, String str3, boolean z16, boolean z17, boolean z18, boolean z19, boolean z20, String[] strArr3, int i11, int[] iArr4, boolean z21, int[] iArr5, boolean z22, String str4, boolean z23, boolean z24) {
        if (iArr == null) {
            this.pli_agreementID = null;
        } else {
            this.pli_agreementID = new int[iArr.length];
            for (int i12 = 0; i12 < iArr.length; i12++) {
                this.pli_agreementID[i12] = iArr[i12];
            }
        }
        this.flg_remote_ui = z9;
        this.pdrID = newString(str);
        this.hriID = newString(str2);
        this.flg_bininfo_settable = z10;
        this.flg_remote_backtoapp = z11;
        this.flg_remote_certificationguide = z12;
        this.sub_model = i9;
        if (strArr == null) {
            this.remoteui_link = null;
        } else {
            this.remoteui_link = new String[strArr.length];
            for (int i13 = 0; i13 < strArr.length; i13++) {
                String[] strArr4 = this.remoteui_link;
                strArr4[i13] = null;
                String str5 = strArr[i13];
                if (str5 != null) {
                    strArr4[i13] = new String(str5);
                }
            }
        }
        if (strArr2 == null) {
            this.remoteui_link_typea = null;
        } else {
            this.remoteui_link_typea = new String[strArr2.length];
            for (int i14 = 0; i14 < strArr2.length; i14++) {
                String[] strArr5 = this.remoteui_link_typea;
                strArr5[i14] = null;
                String str6 = strArr2[i14];
                if (str6 != null) {
                    strArr5[i14] = new String(str6);
                }
            }
        }
        this.flg_web_manual = z13;
        this.flg_network_settings = z14;
        if (iArr2 == null) {
            this.webservice_agreement = null;
        } else {
            this.webservice_agreement = new int[iArr2.length];
            for (int i15 = 0; i15 < iArr2.length; i15++) {
                this.webservice_agreement[i15] = iArr2[i15];
            }
        }
        if (iArr3 == null) {
            this.questionnaire_state = null;
        } else {
            this.questionnaire_state = new int[iArr3.length];
            for (int i16 = 0; i16 < iArr3.length; i16++) {
                this.questionnaire_state[i16] = iArr3[i16];
            }
        }
        this.flg_hdd = z15;
        this.box_num = i10;
        this.product_serialnumber = str3;
        this.flg_wireless_connect_button = z16;
        this.flg_showable_webview = z17;
        this.flg_password_skip_wifi = z18;
        this.flg_communication_bar = z19;
        this.flg_blemode_autolaunch_directmode = z20;
        if (strArr3 == null) {
            this.remoteui_link_typeb = null;
        } else {
            this.remoteui_link_typeb = new String[strArr3.length];
            for (int i17 = 0; i17 < strArr3.length; i17++) {
                String[] strArr6 = this.remoteui_link_typeb;
                strArr6[i17] = null;
                String str7 = strArr3[i17];
                if (str7 != null) {
                    strArr6[i17] = new String(str7);
                }
            }
        }
        this.password_skip_wifi_type = i11;
        if (iArr4 == null) {
            this.mqtt_connection = null;
        } else {
            this.mqtt_connection = new int[iArr4.length];
            for (int i18 = 0; i18 < iArr4.length; i18++) {
                this.mqtt_connection[i18] = iArr4[i18];
            }
        }
        this.flg_printer_set_up = z21;
        if (iArr5 == null) {
            this.auto_printer_rom_update = null;
        } else {
            this.auto_printer_rom_update = new int[iArr5.length];
            for (int i19 = 0; i19 < iArr5.length; i19++) {
                this.auto_printer_rom_update[i19] = iArr5[i19];
            }
        }
        this.flg_printer_subscription_status = z22;
        this.cpc_version = str4;
        this.flg_set_configuration_without_start_job = z23;
        this.flg_reduce_ink_smudges = z24;
    }

    public void setProductSerialnumber(String str) {
        this.product_serialnumber = str;
        encryptProductSerialnumberIfNeeded();
    }
}
