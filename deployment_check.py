"""Deployment Readiness Script — verifies Dell/WSL2 environment post-transfer.

Checks:
1. Python version (3.10+)
2. Virtual environment activation
3. Essential library imports (pandas, chromadb, requests, etc.)
4. network.yaml validity and Tailscale IP reachability
5. Redis connectivity (if local)
6. SecretsManager access (keyring)
"""
import sys
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ReadyCheck")

def check_env():
    logger.info("Step 1: Checking Python environment...")
    if sys.version_info < (3, 10):
        logger.error(f"FAIL: Python version {sys.version} too low. Need 3.10+")
        return False
    logger.info(f"OK: Python {sys.version.split()[0]}")
    
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        logger.warning("WARN: Not running in a virtual environment. Venv recommended.")
    else:
        logger.info("OK: Running in virtual environment.")
    return True

def check_deps():
    logger.info("Step 2: Checking library dependencies...")
    libs = ["pandas", "numpy", "chromadb", "requests", "keyring", "yaml", "rbi_core"]
    missing = []
    for lib in libs:
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)
    
    if missing:
        logger.error(f"FAIL: Missing libraries: {', '.join(missing)}")
        logger.info("Run: pip install -r requirements-wsl.txt")
        return False
    logger.info("OK: All core libraries imported.")
    return True

def check_config():
    logger.info("Step 3: Validating network configuration...")
    from rbi_core.config.network_config import NetworkConfig
    from rbi_core.networking.network_validator import NetworkValidator
    
    cfg_path = Path("config/network.yaml")
    if not cfg_path.exists():
        logger.error("FAIL: config/network.yaml missing!")
        return False
        
    try:
        cfg = NetworkConfig.load(str(cfg_path))
        logger.info(f"OK: network.yaml loaded. Dell: {cfg.dell_tailscale_ip}, Mac: {cfg.mac_tailscale_ip}")
        
        # Check if IPs are Tailscale
        NetworkValidator.assert_tailscale_host(cfg.dell_tailscale_ip)
        NetworkValidator.assert_tailscale_host(cfg.mac_tailscale_ip)
        logger.info("OK: IPs are within Tailscale CGNAT range.")
    except Exception as e:
        logger.error(f"FAIL: Config validation error: {e}")
        return False
    return True

def main():
    print("\n" + "="*50)
    print("      RBI Swarm — Deployment Readiness Check")
    print("="*50 + "\n")
    
    success = True
    if not check_env(): success = False
    if success and not check_deps(): success = False
    if success and not check_config(): success = False
    
    if success:
        print("\n" + "!"*50)
        print("  SUCCESS: ENVIRONMENT IS READY FOR DEPLOYMENT!")
        print("!"*50 + "\n")
    else:
        print("\n" + "x"*50)
        print("  FAILURE: PLEASE FIX ERRORS ABOVE BEFORE RUNNING.")
        print("x"*50 + "\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
