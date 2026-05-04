
#include "model_decryptor.hpp"
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <openssl/hmac.h>
#include <random>
#include <openssl/err.h>
#include <iostream>
#include <sstream>
#include <regex>


ModelDecryptor::ModelDecryptor(const std::string& key_path) {

    try {
        key_ = readKey(key_path);
        if (key_.size() != KEY_SIZE) {
            throw std::runtime_error("Invalid key size: expected 32 bytes");
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to initialize decryptor: " + std::string(e.what()));
    }
}





ModelDecryptor::~ModelDecryptor() {
    // Clean up temporary directories
    for (const auto& dir : temp_directories_) {
        try {
            if (std::filesystem::exists(dir)) {
                std::filesystem::remove_all(dir);
                std::cout << "Cleaned up temporary directory: " << dir << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to clean up directory " << dir << ": " << e.what() << std::endl;
        }
    }
    OPENSSL_cleanse(key_.data(), key_.size());
}


std::vector<uint8_t> ModelDecryptor::readKey(const std::string& key_path) {
    std::cout << "Reading key from: " << key_path << std::endl;
    
    std::ifstream key_file(key_path, std::ios::binary);
    if (!key_file) {
        throw std::runtime_error("Failed to open key file: " + key_path);
    }
    
    std::vector<uint8_t> key((std::istreambuf_iterator<char>(key_file)),
                            std::istreambuf_iterator<char>());
    
    std::cout << "Read key length: " << key.size() << " bytes" << std::endl;
    std::cout << "First 8 bytes of key: ";
    for (int i = 0; i < std::min(8, static_cast<int>(key.size())); i++) {
        printf("%02x", key[i]);
    }
    std::cout << std::endl;

    if (key.size() != KEY_SIZE) {
        throw std::runtime_error("Invalid key size: expected " + std::to_string(KEY_SIZE) + 
                               " bytes, got " + std::to_string(key.size()) + " bytes");
    }
    return key;
}




std::string ModelDecryptor::createTemporaryDirectory(const std::string& prefix) {
    std::string temp_dir = std::filesystem::temp_directory_path().string() + "/" + prefix +
                          std::to_string(std::random_device{}());
    std::filesystem::create_directory(temp_dir);
    temp_directories_.push_back(temp_dir);  // Track for cleanup
    std::cout << "Created temporary directory: " << temp_dir << std::endl;
    return temp_dir;
}

void ModelDecryptor::updateXmlWeightsPath(const std::string& xml_path, const std::string& weights_path) {
    std::cout << "Updating weights reference in XML file: " << xml_path << std::endl;
    
    // Read XML content
    std::ifstream xml_file(xml_path);
    std::stringstream buffer;
    buffer << xml_file.rdbuf();
    std::string content = buffer.str();
    xml_file.close();

    // Create relative path for weights file
    std::filesystem::path xml_dir = std::filesystem::path(xml_path).parent_path();
    std::filesystem::path weights = std::filesystem::path(weights_path);
    std::string relative_weights = std::filesystem::relative(weights, xml_dir).string();

    // Update weights reference in XML
    std::regex weights_pattern("weights=\"[^\"]*\"");
    std::string new_weights = "weights=\"" + relative_weights + "\"";
    content = std::regex_replace(content, weights_pattern, new_weights);

    // Write updated XML
    std::ofstream out_file(xml_path);
    out_file << content;
    std::cout << "Updated weights path to: " << relative_weights << std::endl;
}







std::vector<uint8_t> ModelDecryptor::decryptFile(const std::string& encrypted_path) {
    // 1. Read the encrypted file
    std::ifstream file(encrypted_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open encrypted file: " + encrypted_path);
    }

    // 2. Read IV (16 bytes)
    std::vector<uint8_t> iv(IV_SIZE);
    file.read(reinterpret_cast<char*>(iv.data()), IV_SIZE);
    if (!file) {
        throw std::runtime_error("Failed to read IV from file");
    }

    // 3. Read stored HMAC (32 bytes for SHA256)
    std::vector<uint8_t> stored_hmac(HMAC_SIZE);
    file.read(reinterpret_cast<char*>(stored_hmac.data()), HMAC_SIZE);
    if (!file) {
        throw std::runtime_error("Failed to read HMAC from file");
    }

    // 4. Read encrypted data
    std::vector<uint8_t> encrypted_data(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());

    // 5. Decrypt the data first
    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    if (!ctx) {
        throw std::runtime_error("Failed to create cipher context");
    }

    std::vector<uint8_t> decrypted_data(encrypted_data.size() + EVP_MAX_BLOCK_LENGTH);
    int decrypted_len = 0;
    int final_len = 0;

    try {
        // Initialize decryption
        if (EVP_DecryptInit_ex(ctx, EVP_aes_256_cbc(), nullptr, key_.data(), iv.data()) != 1) {
            throw std::runtime_error("Failed to initialize decryption");
        }

        // Perform decryption
        if (EVP_DecryptUpdate(ctx, decrypted_data.data(), &decrypted_len,
                             encrypted_data.data(), encrypted_data.size()) != 1) {
            throw std::runtime_error("Decryption failed");
        }

        // Finalize decryption
        if (EVP_DecryptFinal_ex(ctx, decrypted_data.data() + decrypted_len, &final_len) != 1) {
            throw std::runtime_error("Failed to finalize decryption");
        }

        // Resize to actual decrypted size
        decrypted_data.resize(decrypted_len + final_len);

        // 6. Verify HMAC on the decrypted data
        unsigned char calculated_hmac[EVP_MAX_MD_SIZE];
        unsigned int hmac_len;
        
        HMAC(EVP_sha256(), key_.data(), key_.size(),
             decrypted_data.data(), decrypted_data.size(),
             calculated_hmac, &hmac_len);

        if (hmac_len != stored_hmac.size() || 
            CRYPTO_memcmp(calculated_hmac, stored_hmac.data(), hmac_len) != 0) {
            throw std::runtime_error("HMAC verification failed");
        }

        EVP_CIPHER_CTX_free(ctx);
        return decrypted_data;
    }
    catch (const std::exception& e) {
        EVP_CIPHER_CTX_free(ctx);
        throw;
    }
}

ModelDecryptor::DecryptedPaths ModelDecryptor::decryptModelFiles(const std::string& model_path) {
    try {
        std::cout << "\nDecrypting model files..." << std::endl;
        std::cout << "Source model path: " << model_path << std::endl;
        
        // Create temporary directory
        std::string temp_dir = createTemporaryDirectory();
        
        // Setup paths
        std::filesystem::path path(model_path);
        std::string base_name = path.filename().string();
        std::string weights_name = path.stem().string() + ".bin";
        
        std::string temp_model = temp_dir + "/" + base_name;
        std::string temp_weights = temp_dir + "/" + weights_name;
        
        std::cout << "Decrypting model to: " << temp_model << std::endl;
        std::cout << "Decrypting weights to: " << temp_weights << std::endl;
        
        // Decrypt files
        auto model_data = decryptFile(model_path + ".enc");
        writeToTempFile(model_data, temp_model);
        
        std::string weights_path = model_path.substr(0, model_path.size() - 3) + "bin";
        auto weights_data = decryptFile(weights_path + ".enc");
        writeToTempFile(weights_data, temp_weights);
        
        // Update XML to reference correct weights path
        updateXmlWeightsPath(temp_model, temp_weights);
        
        // Verify files exist and are readable
        if (!std::filesystem::exists(temp_model)) {
            throw std::runtime_error("Decrypted model file not found: " + temp_model);
        }
        if (!std::filesystem::exists(temp_weights)) {
            throw std::runtime_error("Decrypted weights file not found: " + temp_weights);
        }
        
        std::cout << "Model file size: " << std::filesystem::file_size(temp_model) << " bytes" << std::endl;
        std::cout << "Weights file size: " << std::filesystem::file_size(temp_weights) << " bytes" << std::endl;
        
        return {temp_model, temp_weights, temp_dir};
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to decrypt model files: " + std::string(e.what()));
    }
}



void ModelDecryptor::verifyHMAC(const std::vector<uint8_t>& data, 
                               const std::vector<uint8_t>& stored_hmac) {
    unsigned char hmac[EVP_MAX_MD_SIZE];
    unsigned int hmac_len;

    HMAC(EVP_sha256(), key_.data(), key_.size(),
         data.data(), data.size(),
         hmac, &hmac_len);

    std::cout << "Calculated HMAC length: " << hmac_len << std::endl;
    std::cout << "Stored HMAC length: " << stored_hmac.size() << std::endl;

    // Print first few bytes of both HMACs for comparison
    std::cout << "Calculated HMAC (first 8 bytes): ";
    for (int i = 0; i < 8; i++) {
        printf("%02x", hmac[i]);
    }
    std::cout << std::endl;

    std::cout << "Stored HMAC (first 8 bytes): ";
    for (int i = 0; i < 8; i++) {
        printf("%02x", stored_hmac[i]);
    }
    std::cout << std::endl;

    if (hmac_len != stored_hmac.size() || 
        CRYPTO_memcmp(hmac, stored_hmac.data(), hmac_len) != 0) {
        throw std::runtime_error("HMAC verification failed");
    }
}



void ModelDecryptor::writeToTempFile(const std::vector<uint8_t>& data, 
    const std::string& temp_path) {
    std::cout << "Writing " << data.size() << " bytes to " << temp_path << std::endl;
    
    std::ofstream temp_file(temp_path, std::ios::binary);
    if (!temp_file) {
        throw std::runtime_error("Failed to create temporary file: " + temp_path);
    }
    
    temp_file.write(reinterpret_cast<const char*>(data.data()), data.size());
    
    // Check if this is an XML file by examining the file extension
    // We use string operations that work in C++17 and earlier
    std::string extension = temp_path.substr(temp_path.find_last_of(".") + 1);
    if (extension == "xml") {
        std::ifstream check_file(temp_path);
        std::string line;
        std::cout << "Checking XML file for weights reference..." << std::endl;
        while (std::getline(check_file, line)) {
            if (line.find("weights=") != std::string::npos) {
                std::cout << "Found weights reference: " << line << std::endl;
                break;
            }
        }
    }
}


