#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>

#include <openssl/aes.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/evperr.h>
#include <openssl/aes.h>
#include <openssl/crypto.h>

static std::vector<uint8_t> str_to_bytes(const std::string &message) {
    std::vector<uint8_t> out(message.size());
    for (size_t n = 0; n < message.size(); n++) {
        out[n] = message[n];
    }
    return out;
}

static std::string bytes_to_str(const std::vector<uint8_t> &bytes) {
    return std::string(bytes.begin(), bytes.end());
}

bool base64_decode(const std::string &encoded_input,
                   std::vector<uint8_t> &decoded_output,
                   bool with_newlines = false) {
    BIO *b64_bio = BIO_new(BIO_f_base64());
    BIO *mem_bio = BIO_new_mem_buf(encoded_input.data(), encoded_input.size());
    BIO_push(b64_bio, mem_bio);
    if (!with_newlines) {
        BIO_set_flags(b64_bio, BIO_FLAGS_BASE64_NO_NL);
    }
    decoded_output.resize(encoded_input.size());
    int decoded_len = BIO_read(b64_bio, decoded_output.data(), encoded_input.size());
    if (decoded_len <= 0) {
        BIO_free_all(b64_bio);
        return false;
    }
    decoded_output.resize(decoded_len);
    BIO_free_all(b64_bio);
    return true;
}

static void print_hex(const std::vector<uint8_t> &data) {
    for (size_t i = 0; i < data.size(); ++i) {
        std::cout << std::hex << std::uppercase << (int)data[i];
    }
    std::cout << std::dec << std::endl;
}

bool aes_decrypt(const std::vector<uint8_t> &message,
                 const std::vector<uint8_t> &key,
                 const std::vector<uint8_t> &iv,
                 std::vector<uint8_t> &output) {
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx)
        return false;
    if (1 != EVP_DecryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, &key[0], &iv[0])) {
        EVP_CIPHER_CTX_free(ctx);
        return false;
    }
    output.resize(message.size());
    int len;
    int plaintext_len;
    if (1 != EVP_DecryptUpdate(ctx, &output[0], &len, &message[0], message.size())) {
        EVP_CIPHER_CTX_free(ctx);
        return false;
    }
    plaintext_len = len;
    if (1 != EVP_DecryptFinal_ex(ctx, &output[len], &len)) {
        EVP_CIPHER_CTX_free(ctx);
        return false;
    }
    plaintext_len += len;
    output.resize(plaintext_len);
    EVP_CIPHER_CTX_free(ctx);
    return true;
}

int main(int, char **) {
    const std::string key = "passwordpasswordpasswordpassword";
    const std::string iv = "1234567890123456";
    const std::string message = "hello world";
    print_hex(str_to_bytes(key));
    print_hex(str_to_bytes(iv));
    std::vector<uint8_t> enc_data; // echo -n "hello world" | openssl enc -aes-256-cbc -K "70617373776F726470617373776F726470617373776F726470617373776F7264" -iv "31323334353637383930313233343536" -a
    if (!base64_decode("bS+CwTtz7flIAAmc5314yg==", enc_data)) {
        std::cerr << "Failed to decode" << std::endl;
        return 1;
    }
    std::vector<uint8_t> dec_data;
    if (!aes_decrypt(enc_data, str_to_bytes(key), str_to_bytes(iv), dec_data)) {
        std::cerr << "Failed to decrypt" << std::endl;
        return 1;
    }
    std::cout << bytes_to_str(dec_data) << std::endl;
    return 0;
}
