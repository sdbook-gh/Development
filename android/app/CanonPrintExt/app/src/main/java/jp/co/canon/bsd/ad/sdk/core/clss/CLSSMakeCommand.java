package jp.co.canon.bsd.ad.sdk.core.clss;

import jp.co.canon.bsd.ad.sdk.core.clss.struct.CLSSCancelJobParam;
import jp.co.canon.bsd.ad.sdk.core.clss.struct.CLSSEndJobParam;
import jp.co.canon.bsd.ad.sdk.core.clss.struct.CLSSGetStatusParam;
import jp.co.canon.bsd.ad.sdk.core.clss.struct.CLSSModeShiftParam;
import jp.co.canon.bsd.ad.sdk.core.clss.struct.CLSSSendDataParam;
import jp.co.canon.bsd.ad.sdk.core.clss.struct.CLSSSetConfigurationParam;
import jp.co.canon.bsd.ad.sdk.core.clss.struct.CLSSSetConfigurationWithoutStartJobParam;
import jp.co.canon.bsd.ad.sdk.core.clss.struct.CLSSSetJobConfigurationParam;
import jp.co.canon.bsd.ad.sdk.core.clss.struct.CLSSSetPageConfigurationParam;
import jp.co.canon.bsd.ad.sdk.core.clss.struct.CLSSStartJobParam;

/* loaded from: /mnt/f/print/classes3.dex — getters cleaned for Java definite-return rules */
public class CLSSMakeCommand {
    private String ivec_command;
    private String str_error = "load library Error( nothing code \" System.loadLibrary();\" or nothing JNI folder)";
    private String str_error2 = "Error = ";

    public native int WrapperCLSSMakeCommandFRomUpModeNew();

    public native int WrapperCLSSMakeCommandGetCancelJobNew(CLSSCancelJobParam cLSSCancelJobParam);

    public native int WrapperCLSSMakeCommandGetCapabilityNew(int i9);

    public native int WrapperCLSSMakeCommandGetConfigurationNew(int i9);

    public native int WrapperCLSSMakeCommandGetEndJobNew(CLSSEndJobParam cLSSEndJobParam);

    public native int WrapperCLSSMakeCommandGetModeShiftNew(CLSSModeShiftParam cLSSModeShiftParam);

    public native int WrapperCLSSMakeCommandGetStatusNew(CLSSGetStatusParam cLSSGetStatusParam);

    public native int WrapperCLSSMakeCommandHandoveroffNew();

    public native int WrapperCLSSMakeCommandPowerOffNew();

    public native int WrapperCLSSMakeCommandResumeErrorNew();

    public native int WrapperCLSSMakeCommandRunConnectivityCheckNew();

    public native int WrapperCLSSMakeCommandSendDataNew(CLSSSendDataParam cLSSSendDataParam);

    public native int WrapperCLSSMakeCommandSetConfigurationNew(CLSSSetConfigurationParam cLSSSetConfigurationParam, int i9, String str);

    public native int WrapperCLSSMakeCommandSetConfigurationWithoutStartJob(CLSSSetConfigurationWithoutStartJobParam cLSSSetConfigurationWithoutStartJobParam);

    public native int WrapperCLSSMakeCommandSetJobConfigurationNew(CLSSSetJobConfigurationParam cLSSSetJobConfigurationParam, String str);

    public native int WrapperCLSSMakeCommandSetPageConfigurationNew(CLSSSetPageConfigurationParam cLSSSetPageConfigurationParam);

    public native int WrapperCLSSMakeCommandStartJobNew(CLSSStartJobParam cLSSStartJobParam);

    private String finish(int rc) {
        if (rc >= 0) {
            return this.ivec_command;
        }
        this.ivec_command = null;
        throw new CLSS_Exception(ClssErr.c(rc, this.str_error2, new StringBuilder()));
    }

    public String getCancelJob(CLSSCancelJobParam cLSSCancelJobParam) {
        int rc;
        try {
            rc = WrapperCLSSMakeCommandGetCancelJobNew(cLSSCancelJobParam);
        } catch (UnsatisfiedLinkError unused) {
            throw new CLSS_Exception(this.str_error);
        } catch (Exception unused) {
            rc = -3;
        }
        return finish(rc);
    }

    public String getEndJob(CLSSEndJobParam cLSSEndJobParam) {
        int rc;
        try {
            rc = WrapperCLSSMakeCommandGetEndJobNew(cLSSEndJobParam);
        } catch (UnsatisfiedLinkError unused) {
            throw new CLSS_Exception(this.str_error);
        } catch (Exception unused) {
            rc = -3;
        }
        return finish(rc);
    }

    public String getFRomUpMode() {
        int rc;
        try {
            rc = WrapperCLSSMakeCommandFRomUpModeNew();
        } catch (UnsatisfiedLinkError unused) {
            throw new CLSS_Exception(this.str_error);
        } catch (Exception unused) {
            rc = -3;
        }
        return finish(rc);
    }

    public String getGetCapability(int i9) {
        int rc;
        try {
            rc = WrapperCLSSMakeCommandGetCapabilityNew(i9);
        } catch (UnsatisfiedLinkError unused) {
            throw new CLSS_Exception(this.str_error);
        } catch (Exception unused) {
            rc = -3;
        }
        return finish(rc);
    }

    public String getGetConfigration(int i9) {
        int rc;
        try {
            rc = WrapperCLSSMakeCommandGetConfigurationNew(i9);
        } catch (UnsatisfiedLinkError unused) {
            throw new CLSS_Exception(this.str_error);
        } catch (Exception unused) {
            rc = -3;
        }
        return finish(rc);
    }

    public String getGetStatus(CLSSGetStatusParam cLSSGetStatusParam) {
        int rc;
        try {
            rc = WrapperCLSSMakeCommandGetStatusNew(cLSSGetStatusParam);
        } catch (UnsatisfiedLinkError unused) {
            throw new CLSS_Exception(this.str_error);
        } catch (Exception unused) {
            rc = -3;
        }
        return finish(rc);
    }

    public String getHandoveroff() {
        int rc;
        try {
            rc = WrapperCLSSMakeCommandHandoveroffNew();
        } catch (UnsatisfiedLinkError unused) {
            throw new CLSS_Exception(this.str_error);
        } catch (Exception unused) {
            rc = -3;
        }
        return finish(rc);
    }

    public String getModeShift(CLSSModeShiftParam cLSSModeShiftParam) {
        int rc;
        try {
            rc = WrapperCLSSMakeCommandGetModeShiftNew(cLSSModeShiftParam);
        } catch (UnsatisfiedLinkError unused) {
            throw new CLSS_Exception(this.str_error);
        } catch (Exception unused) {
            rc = -3;
        }
        return finish(rc);
    }

    public String getPowerOff() {
        int rc;
        try {
            rc = WrapperCLSSMakeCommandPowerOffNew();
        } catch (UnsatisfiedLinkError unused) {
            throw new CLSS_Exception(this.str_error);
        } catch (Exception unused) {
            rc = -3;
        }
        return finish(rc);
    }

    public String getResumeError() {
        int rc;
        try {
            rc = WrapperCLSSMakeCommandResumeErrorNew();
        } catch (UnsatisfiedLinkError unused) {
            throw new CLSS_Exception(this.str_error);
        } catch (Exception unused) {
            rc = -3;
        }
        return finish(rc);
    }

    public String getRunConnectivityCheck() {
        int rc;
        try {
            rc = WrapperCLSSMakeCommandRunConnectivityCheckNew();
        } catch (UnsatisfiedLinkError unused) {
            throw new CLSS_Exception(this.str_error);
        } catch (Exception unused) {
            rc = -3;
        }
        return finish(rc);
    }

    public String getSendData(CLSSSendDataParam cLSSSendDataParam) {
        int rc;
        try {
            rc = WrapperCLSSMakeCommandSendDataNew(cLSSSendDataParam);
        } catch (UnsatisfiedLinkError unused) {
            throw new CLSS_Exception(this.str_error);
        } catch (Exception unused) {
            rc = -3;
        }
        return finish(rc);
    }

    public String getSetConfiguration(CLSSSetConfigurationParam cLSSSetConfigurationParam, int i9, String str) {
        int rc;
        try {
            rc = WrapperCLSSMakeCommandSetConfigurationNew(cLSSSetConfigurationParam, i9, str);
        } catch (UnsatisfiedLinkError unused) {
            throw new CLSS_Exception(this.str_error);
        } catch (Exception unused) {
            rc = -3;
        }
        return finish(rc);
    }

    public String getSetConfigurationWithoutStartJob(CLSSSetConfigurationWithoutStartJobParam cLSSSetConfigurationWithoutStartJobParam) {
        int rc;
        try {
            rc = WrapperCLSSMakeCommandSetConfigurationWithoutStartJob(cLSSSetConfigurationWithoutStartJobParam);
        } catch (UnsatisfiedLinkError unused) {
            throw new CLSS_Exception(this.str_error);
        } catch (Exception unused) {
            rc = -3;
        }
        return finish(rc);
    }

    public String getSetJobConfiguration(CLSSSetJobConfigurationParam cLSSSetJobConfigurationParam, String str) {
        int rc;
        try {
            rc = WrapperCLSSMakeCommandSetJobConfigurationNew(cLSSSetJobConfigurationParam, str);
        } catch (UnsatisfiedLinkError unused) {
            throw new CLSS_Exception(this.str_error);
        } catch (Exception unused) {
            rc = -3;
        }
        return finish(rc);
    }

    public String getSetPageConfiguration(CLSSSetPageConfigurationParam cLSSSetPageConfigurationParam) {
        int rc;
        try {
            rc = WrapperCLSSMakeCommandSetPageConfigurationNew(cLSSSetPageConfigurationParam);
        } catch (UnsatisfiedLinkError unused) {
            throw new CLSS_Exception(this.str_error);
        } catch (Exception unused) {
            rc = -3;
        }
        return finish(rc);
    }

    public String getStartJob(CLSSStartJobParam cLSSStartJobParam) {
        int rc;
        try {
            rc = WrapperCLSSMakeCommandStartJobNew(cLSSStartJobParam);
        } catch (UnsatisfiedLinkError unused) {
            throw new CLSS_Exception(this.str_error);
        } catch (Exception unused) {
            rc = -3;
        }
        return finish(rc);
    }

    public void set(String str) {
        try {
            this.ivec_command = new String(str);
        } catch (Exception unused) {
            this.ivec_command = null;
        }
    }
}
