"""
Field-level encryption utilities for sensitive data
"""

import base64
import hashlib
import logging
import os
from typing import Any, Dict, Optional, Union

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class FieldEncryption:
    """
    Field-level encryption for sensitive data in compliance pipeline
    """
    
    def __init__(self, key: Optional[bytes] = None, password: Optional[str] = None):
        """
        Initialize field encryption
        
        Args:
            key: Encryption key (32 bytes)
            password: Password to derive key from
        """
        if key:
            self.key = key
        elif password:
            self.key = self._derive_key_from_password(password)
        else:
            # Generate a new key
            self.key = Fernet.generate_key()
        
        self.cipher = Fernet(self.key)
        logger.info("Field encryption initialized")
    
    def _derive_key_from_password(self, password: str, salt: Optional[bytes] = None) -> bytes:
        """Derive encryption key from password"""
        if salt is None:
            salt = b'docbridgeguard_salt_2024'  # Fixed salt for consistency
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def encrypt_field(self, value: Union[str, int, float]) -> str:
        """
        Encrypt a single field value
        
        Args:
            value: Value to encrypt
            
        Returns:
            Base64 encoded encrypted value
        """
        try:
            # Convert value to string
            str_value = str(value)
            
            # Encrypt
            encrypted_bytes = self.cipher.encrypt(str_value.encode('utf-8'))
            
            # Return base64 encoded string
            return base64.urlsafe_b64encode(encrypted_bytes).decode('utf-8')
        
        except Exception as e:
            logger.error(f"Field encryption failed: {e}")
            return f"[ENCRYPTION_ERROR: {str(value)[:10]}...]"
    
    def decrypt_field(self, encrypted_value: str) -> str:
        """
        Decrypt a single field value
        
        Args:
            encrypted_value: Base64 encoded encrypted value
            
        Returns:
            Decrypted value as string
        """
        try:
            # Decode from base64
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode('utf-8'))
            
            # Decrypt
            decrypted_bytes = self.cipher.decrypt(encrypted_bytes)
            
            # Return decoded string
            return decrypted_bytes.decode('utf-8')
        
        except Exception as e:
            logger.error(f"Field decryption failed: {e}")
            return f"[DECRYPTION_ERROR]"
    
    def encrypt_dict(self, data: Dict[str, Any], fields_to_encrypt: list[str]) -> Dict[str, Any]:
        """
        Encrypt specified fields in a dictionary
        
        Args:
            data: Dictionary containing data
            fields_to_encrypt: List of field names to encrypt
            
        Returns:
            Dictionary with encrypted fields
        """
        encrypted_data = data.copy()
        
        for field in fields_to_encrypt:
            if field in encrypted_data:
                encrypted_data[field] = self.encrypt_field(encrypted_data[field])
                encrypted_data[f"{field}_encrypted"] = True
        
        return encrypted_data
    
    def decrypt_dict(self, data: Dict[str, Any], fields_to_decrypt: list[str]) -> Dict[str, Any]:
        """
        Decrypt specified fields in a dictionary
        
        Args:
            data: Dictionary containing encrypted data
            fields_to_decrypt: List of field names to decrypt
            
        Returns:
            Dictionary with decrypted fields
        """
        decrypted_data = data.copy()
        
        for field in fields_to_decrypt:
            if field in decrypted_data and data.get(f"{field}_encrypted", False):
                decrypted_data[field] = self.decrypt_field(decrypted_data[field])
                decrypted_data[f"{field}_encrypted"] = False
        
        return decrypted_data
    
    def hash_field(self, value: Union[str, int, float], algorithm: str = "sha256") -> str:
        """
        Create irreversible hash of field value
        
        Args:
            value: Value to hash
            algorithm: Hash algorithm ('sha256', 'sha512', 'md5')
            
        Returns:
            Hexadecimal hash string
        """
        try:
            str_value = str(value)
            
            if algorithm == "sha256":
                hash_obj = hashlib.sha256(str_value.encode('utf-8'))
            elif algorithm == "sha512":
                hash_obj = hashlib.sha512(str_value.encode('utf-8'))
            elif algorithm == "md5":
                hash_obj = hashlib.md5(str_value.encode('utf-8'))
            else:
                raise ValueError(f"Unsupported hash algorithm: {algorithm}")
            
            return hash_obj.hexdigest()
        
        except Exception as e:
            logger.error(f"Field hashing failed: {e}")
            return f"[HASH_ERROR]"
    
    def create_pseudonymized_id(self, value: Union[str, int, float], prefix: str = "ID") -> str:
        """
        Create a pseudonymized identifier for a value
        
        Args:
            value: Value to pseudonymize
            prefix: Prefix for the pseudonymized ID
            
        Returns:
            Pseudonymized identifier
        """
        try:
            # Create hash of the value
            hash_value = self.hash_field(value, "sha256")
            
            # Take first 8 characters of hash for readability
            short_hash = hash_value[:8].upper()
            
            return f"{prefix}_{short_hash}"
        
        except Exception as e:
            logger.error(f"Pseudonymization failed: {e}")
            return f"{prefix}_ERROR"
    
    def get_key_info(self) -> Dict[str, str]:
        """Get information about the encryption key"""
        return {
            "key_length": str(len(self.key)),
            "key_fingerprint": hashlib.sha256(self.key).hexdigest()[:16],
            "algorithm": "Fernet (AES 128 in CBC mode)"
        }
    
    def export_key(self) -> str:
        """Export encryption key as base64 string"""
        return base64.urlsafe_b64encode(self.key).decode('utf-8')
    
    @classmethod
    def from_key_string(cls, key_string: str) -> 'FieldEncryption':
        """
        Create FieldEncryption instance from base64 key string
        
        Args:
            key_string: Base64 encoded key string
            
        Returns:
            FieldEncryption instance
        """
        try:
            key = base64.urlsafe_b64decode(key_string.encode('utf-8'))
            return cls(key=key)
        except Exception as e:
            logger.error(f"Failed to load key from string: {e}")
            raise ValueError("Invalid key string")
    
    def secure_compare(self, value1: str, value2: str) -> bool:
        """
        Securely compare two values by comparing their hashes
        
        Args:
            value1: First value
            value2: Second value
            
        Returns:
            True if values are equal
        """
        try:
            hash1 = self.hash_field(value1, "sha256")
            hash2 = self.hash_field(value2, "sha256")
            
            # Use constant-time comparison
            return hashlib.compare_digest(hash1, hash2)
        
        except Exception as e:
            logger.error(f"Secure comparison failed: {e}")
            return False