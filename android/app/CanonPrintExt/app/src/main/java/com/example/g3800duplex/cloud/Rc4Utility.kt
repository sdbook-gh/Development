package com.example.g3800duplex.cloud

import java.io.ByteArrayOutputStream
import java.io.OutputStream
import javax.crypto.Cipher
import javax.crypto.CipherOutputStream
import javax.crypto.spec.SecretKeySpec

/**
 * RC4 (ARC4) decrypt helper — same CIPHER_INSTANCE as official EncryptionUtility.
 * Stream cipher: encrypt mode XOR decrypts encrypted bytes.
 */
object Rc4Utility {
    fun decrypt(encrypted: ByteArray, key: String): ByteArray {
        val cipher = Cipher.getInstance("ARC4")
        cipher.init(Cipher.ENCRYPT_MODE, SecretKeySpec(key.toByteArray(Charsets.UTF_8), "RC4"))
        val out = ByteArrayOutputStream(encrypted.size)
        CipherOutputStream(out, cipher).use { cos ->
            cos.write(encrypted)
            cos.flush()
        }
        return out.toByteArray()
    }

    fun decryptTo(encrypted: ByteArray, key: String, output: OutputStream) {
        val cipher = Cipher.getInstance("ARC4")
        cipher.init(Cipher.ENCRYPT_MODE, SecretKeySpec(key.toByteArray(Charsets.UTF_8), "RC4"))
        CipherOutputStream(output, cipher).use { cos ->
            cos.write(encrypted)
            cos.flush()
        }
    }
}
