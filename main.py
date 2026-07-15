# -*- coding: utf-8 -*-
"""
ADAS System Application - Main Entry Point
"""

import sys
import logging
import os
import pickle                # <-- added pickle
from PyQt5.QtWidgets import QApplication
from adas_system import ADASSystem
from gui import ADASApp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('adas_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Starting ADAS System")

        # Initialize ADAS system with config
        adas_system = ADASSystem("config.json")

        # Try to load a previously saved calibration (e.g., from a fixed path)
        calib_path = "calibration.pkl"
        if os.path.exists(calib_path):
            try:
                with open(calib_path, 'rb') as f:
                    calib_data = pickle.load(f)
                    adas_system.load_calibration_data(calib_data)
                    logger.info("Loaded calibration from file")
            except Exception as e:
                logger.error(f"Failed to load calibration: {e}")
        else:
            logger.info("No calibration file found. Using defaults or manual calibration later.")

        # Audio test
        # audio_status = adas_system.get_audio_status()
        # print(f"Audio enabled: {audio_status['enabled']}")
        # print(f"Audio initialized: {audio_status['initialized']}")
        # print(f"Number of sounds loaded: {audio_status['sounds_loaded']}")

        # if audio_status['initialized']:
        #     adas_system.test_audio_system()
        # else:
        #     print("WARNING: Audio system not initialized.")

        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        window = ADASApp(adas_system)
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        logger.error(f"Error starting ADAS system: {e}")
        raise

if __name__ == "__main__":
    main()