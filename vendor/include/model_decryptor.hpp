

#pragma once
#include <vector>
#include <string>
#include <memory>
#include <openssl/evp.h>
#include <openssl/aes.h>



// Class responsible for decrypting OpenVINO model files at runtime

class ModelDecryptor {

public:
    // Constructor takes path to encryption key file


    struct DecryptedPaths {
        std::string model_path;    // Path to decrypted .xml file
        std::string weights_path;  // Path to decrypted .bin file
        std::string temp_dir;      // Path to temporary directory containing the files

    };

    explicit ModelDecryptor(const std::string& key_path);

    ~ModelDecryptor();

    
    DecryptedPaths decryptModelFiles(const std::string& model_path);


private:


    
    // Constants for encryption parameters
    static constexpr size_t IV_SIZE = 16;      // AES block size
    static constexpr size_t HMAC_SIZE = 32;    // SHA-256 hash size
    static constexpr size_t KEY_SIZE = 32;     // AES-256 key size


// Member variables
    std::vector<uint8_t> key_;         
            // Encryption key
    std::vector<std::string> temp_directories_;// Track temporary directories for cleanup


// Helper methods
    std::vector<uint8_t> readKey(const std::string& key_path);
    std::vector<uint8_t> decryptFile(const std::string& encrypted_path);
    void verifyHMAC(const std::vector<uint8_t>& data, const std::vector<uint8_t>& hmac);
    void writeToTempFile(const std::vector<uint8_t>& data, const std::string& temp_path);
    std::string createTemporaryDirectory(const std::string& prefix = "model_");
    void updateXmlWeightsPath(const std::string& xml_path, const std::string& weights_path);
};


   
