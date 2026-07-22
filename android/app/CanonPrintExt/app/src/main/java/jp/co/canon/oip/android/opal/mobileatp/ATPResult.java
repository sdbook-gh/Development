package jp.co.canon.oip.android.opal.mobileatp;

import jp.co.canon.oip.android.opal.mobileatp.error.ATPException;
import jp.co.canon.oip.android.opal.mobileatp.util.g;

/* loaded from: /mnt/f/print/classes3.dex */
public class ATPResult {
    public static final int ERR_FUNC_TYPE_ACCESS_TOKEN = 2;
    public static final int ERR_FUNC_TYPE_DEVICE_REGISTRATION = 1;
    public static final int ERR_FUNC_TYPE_NONE = 0;
    public static final int ERR_FUNC_TYPE_SUBSET_TOKEN = 3;
    public static final int RESULT_CODE_NG_ANALYSE_RESPONSE_MESSAGE = 902;
    public static final int RESULT_CODE_NG_AUTHORIZATION = 103;
    public static final int RESULT_CODE_NG_CAMS_ACCESS = 699;
    public static final int RESULT_CODE_NG_CLIENT_ID = 104;
    public static final int RESULT_CODE_NG_CREATE_REQUEST_MESSAGE = 800;
    public static final int RESULT_CODE_NG_CREATE_REQUEST_URLENCODE = 803;
    public static final int RESULT_CODE_NG_CREATE_SERIALNUMBER = 105;
    public static final int RESULT_CODE_NG_CRYPTO_BAD_PADDING = 408;
    public static final int RESULT_CODE_NG_CRYPTO_BLOCK_SIZE = 407;
    public static final int RESULT_CODE_NG_CRYPTO_ERR = 400;
    public static final int RESULT_CODE_NG_CRYPTO_INVALID_ALGORITHM = 401;
    public static final int RESULT_CODE_NG_CRYPTO_INVALID_ALGORITHM_PARAM = 406;
    public static final int RESULT_CODE_NG_CRYPTO_INVALID_KEY = 405;
    public static final int RESULT_CODE_NG_CRYPTO_INVALID_KEY_SPEC = 403;
    public static final int RESULT_CODE_NG_CRYPTO_NO_SUCH_ALGORITHM = 402;
    public static final int RESULT_CODE_NG_CRYPTO_NO_SUCH_PADDING = 404;
    public static final int RESULT_CODE_NG_DECRYPT_BAD_PADDING = 506;
    public static final int RESULT_CODE_NG_DECRYPT_BLOCK_SIZE = 505;
    public static final int RESULT_CODE_NG_DECRYPT_CLIENT_CERT = 507;
    public static final int RESULT_CODE_NG_DECRYPT_CLIENT_CERT_PASSWORD = 508;
    public static final int RESULT_CODE_NG_DECRYPT_ERR = 500;
    public static final int RESULT_CODE_NG_DECRYPT_INVALID_ALGORITHM_PARAM = 504;
    public static final int RESULT_CODE_NG_DECRYPT_INVALID_KEY = 503;
    public static final int RESULT_CODE_NG_DECRYPT_KEY = 509;
    public static final int RESULT_CODE_NG_DECRYPT_NO_SUCH_ALGORITHM = 501;
    public static final int RESULT_CODE_NG_DECRYPT_NO_SUCH_PADDING = 502;
    public static final int RESULT_CODE_NG_DEVICE_CREDENTIAL = 101;
    public static final int RESULT_CODE_NG_ENCODE_JSON = 700;
    public static final int RESULT_CODE_NG_ENCODE_JSON_UNSUPPORTED_ENCODE = 701;
    public static final int RESULT_CODE_NG_FILE = 1099;
    public static final int RESULT_CODE_NG_FILE_OPEN = 1002;
    public static final int RESULT_CODE_NG_FILE_READ = 1003;
    public static final int RESULT_CODE_NG_FILE_VALIDATE = 1006;
    public static final int RESULT_CODE_NG_FILE_WRITE = 1004;
    public static final int RESULT_CODE_NG_HTTP = 1;
    public static final int RESULT_CODE_NG_HTTP_NONE_ERROR_CODE = 2;
    public static final int RESULT_CODE_NG_INPUT_PARAM = 100;
    public static final int RESULT_CODE_NG_INTERNAL_FATAL = 102;
    public static final int RESULT_CODE_NG_INVALID_DIRECTORY = 1000;
    public static final int RESULT_CODE_NG_INVALID_FILENAME = 1001;
    public static final int RESULT_CODE_NG_JSON = 799;
    public static final int RESULT_CODE_NG_KEYSTORE_CERTIFICATE = 203;
    public static final int RESULT_CODE_NG_KEYSTORE_ERR = 200;
    public static final int RESULT_CODE_NG_KEYSTORE_IO_EXCEPTION = 204;
    public static final int RESULT_CODE_NG_KEYSTORE_NO_SUCH_ALGORITHM = 202;
    public static final int RESULT_CODE_NG_KEYSTORE_NO_SUCH_PROVIDER = 201;
    public static final int RESULT_CODE_NG_NO_WRITE_DATA = 1005;
    public static final int RESULT_CODE_NG_PARSE_JSON = 702;
    public static final int RESULT_CODE_NG_PARSE_PROPERTY = 106;
    public static final int RESULT_CODE_NG_REQUEST_CLIENT_PROTOCOL = 801;
    public static final int RESULT_CODE_NG_REQUEST_CONNECT_TIMEOUT = 805;
    public static final int RESULT_CODE_NG_REQUEST_IO_EXCEPTION = 802;
    public static final int RESULT_CODE_NG_REQUEST_UNSUPPORTED_ENCODE = 804;
    public static final int RESULT_CODE_NG_RESPONSE_IO_EXCEPTION = 901;
    public static final int RESULT_CODE_NG_RESPONSE_PARSE = 900;
    public static final int RESULT_CODE_NG_SSLSOCKET_KEYMNG = 300;
    public static final int RESULT_CODE_NG_SSLSOCKET_KEY_STORE = 303;
    public static final int RESULT_CODE_NG_SSLSOCKET_NO_SUCH_ALGORITHM = 302;
    public static final int RESULT_CODE_NG_SSLSOCKET_UNRECOVERABLE_KEY = 301;
    public static final int RESULT_CODE_NO_USER_AGENT = 107;
    public static final int RESULT_CODE_OK = 0;

    /* renamed from: a, reason: collision with root package name */
    protected int f16823a;

    /* renamed from: b, reason: collision with root package name */
    protected int f16824b;

    /* renamed from: c, reason: collision with root package name */
    protected String f16825c;

    /* renamed from: d, reason: collision with root package name */
    protected String f16826d;

    public ATPResult(int i9, int i10, String str) {
        this.f16823a = i9;
        this.f16824b = i10;
        this.f16825c = str;
    }

    private void a(ATPException aTPException) {
        String str;
        int httpStatusCode;
        String errorDescription;
        int status = aTPException.getStatus();
        str = "";
        if (status == 1 || status == 2) {
            httpStatusCode = aTPException.getHttpStatusCode();
            String errorCode = aTPException.getErrorCode();
            str = g.a(errorCode) ? "" : errorCode;
            errorDescription = aTPException.getErrorDescription();
        } else {
            String message = aTPException.getMessage();
            httpStatusCode = 0;
            if (g.a(message)) {
                errorDescription = "";
            } else {
                str = message;
                errorDescription = "";
            }
        }
        this.f16823a = status;
        this.f16824b = httpStatusCode;
        this.f16825c = str;
        this.f16826d = errorDescription;
    }

    public String getErrorCode() {
        return this.f16825c;
    }

    public String getErrorDescription() {
        return this.f16826d;
    }

    public int getHttpStatusCode() {
        return this.f16824b;
    }

    public int getResultCode() {
        return this.f16823a;
    }

    public ATPResult(ATPException aTPException) {
        a(aTPException);
    }
}
