# Copyright 2019 Gehtsoft USA LLC
#
# Licensed under the license derived from the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License at
#
# http://fxcodebase.com/licenses/open-source/license.html
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import configparser
import os


class ConfigLoader:
    """Helper class to load credentials from config.ini file"""
    
    def __init__(self, config_file='config.ini'):
        """
        Initialize the config loader
        
        Args:
            config_file: Path to the config file (default: 'config.ini')
        """
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_file)
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        self.config.read(config_path)
    
    def get_credentials(self):
        """
        Load credentials from config file
        
        Returns:
            dict: Dictionary containing login, password, url, connection, session, and pin
        """
        credentials = {}
        
        if 'Credentials' not in self.config:
            raise ValueError("'Credentials' section not found in config file")
        
        cred_section = self.config['Credentials']
        
        # Required fields
        credentials['login'] = cred_section.get('login', '').strip()
        credentials['password'] = cred_section.get('password', '').strip()
        credentials['url'] = cred_section.get('url', '').strip()
        credentials['connection'] = cred_section.get('connection', 'Demo').strip()
        
        # Optional fields
        credentials['session'] = cred_section.get('session', '').strip() or None
        credentials['pin'] = cred_section.get('pin', '').strip() or None
        
        # Validate required fields
        if not credentials['login']:
            raise ValueError("'login' is required in config file")
        if not credentials['password']:
            raise ValueError("'password' is required in config file")
        if not credentials['url']:
            raise ValueError("'url' is required in config file")
        
        return credentials
    
    @staticmethod
    def load():
        """
        Convenience method to load credentials
        
        Returns:
            dict: Dictionary containing credentials
        """
        loader = ConfigLoader()
        return loader.get_credentials()


# Example usage
if __name__ == "__main__":
    try:
        creds = ConfigLoader.load()
        print("Credentials loaded successfully:")
        print(f"Login: {creds['login']}")
        print(f"URL: {creds['url']}")
        print(f"Connection: {creds['connection']}")
        if creds['session']:
            print(f"Session: {creds['session']}")
        if creds['pin']:
            print(f"Pin: {'*' * len(creds['pin'])}")
    except Exception as e:
        print(f"Error loading credentials: {e}")

