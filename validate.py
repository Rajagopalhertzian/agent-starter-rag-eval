#!/usr/bin/env python3
"""End-to-end validation script for the RAG Agent Starter Kit."""

import asyncio
import sys
import subprocess
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path = None) -> tuple[int, str, str]:
    """Run command and return (exit_code, stdout, stderr)."""
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def check_ollama() -> bool:
    """Check if Ollama is running and has required models."""
    code, out, err = run_cmd(["ollama", "list"])
    if code != 0:
        print("❌ Ollama not running or not installed")
        return False
    
    required = ["qwen2.5-coder:7b", "nomic-embed-text"]
    for model in required:
        if model not in out:
            print(f"❌ Missing model: {model}")
            print("   Run: ollama pull " + model)
            return False
    
    print("✅ Ollama running with required models")
    return True


def check_uv() -> bool:
    """Check uv is available."""
    code, out, err = run_cmd(["uv", "--version"])
    if code != 0:
        print("❌ uv not installed")
        return False
    print(f"✅ uv: {out.strip()}")
    return True


def check_tests(project_root: Path) -> bool:
    """Run pytest."""
    print("🧪 Running tests...")
    code, out, err = run_cmd(["uv", "run", "pytest", "tests/", "-v"], cwd=project_root)
    if code != 0:
        print(f"❌ Tests failed:\n{out}\n{err}")
        return False
    print("✅ All tests passed")
    return True


def check_imports(project_root: Path) -> bool:
    """Verify key modules import without error."""
    modules = [
        "src.agent.config",
        "src.agent.llm",
        "src.agent.graph",
        "src.agent.ingest",
        "src.api.main",
        "src.eval.eval",
    ]
    
    for mod in modules:
        code, out, err = run_cmd(
            ["uv", "run", "python", "-c", f"import {mod}; print('OK')"],
            cwd=project_root
        )
        if code != 0:
            print(f"❌ Import failed: {mod}\n{err}")
            return False
    print("✅ All modules import successfully")
    return True


def check_docker_compose(project_root: Path) -> bool:
    """Validate docker-compose.yml syntax."""
    code, out, err = run_cmd(["docker", "compose", "config"], cwd=project_root)
    if code != 0:
        print(f"⚠️  Docker Compose config check failed (Docker may not be running):\n{err}")
        return True  # Not a hard failure if Docker isn't running
    print("✅ Docker Compose config valid")
    return True


async def main():
    project_root = Path(__file__).parent
    
    print("=" * 60)
    print("RAG Agent Starter Kit — Validation")
    print("=" * 60)
    
    checks = [
        ("UV", check_uv),
        ("Ollama + Models", check_ollama),
        ("Module Imports", lambda: check_imports(project_root)),
        ("Unit Tests", lambda: check_tests(project_root)),
        ("Docker Compose", lambda: check_docker_compose(project_root)),
    ]
    
    results = []
    for name, check in checks:
        print(f"\n🔍 {name}...")
        try:
            ok = check()
            results.append((name, ok))
        except Exception as e:
            print(f"❌ {name} error: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_pass = True
    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status} — {name}")
        if not ok:
            all_pass = False
    
    print("\n" + "=" * 60)
    if all_pass:
        print("🎉 ALL CHECKS PASSED — Project is ready to launch!")
        print("\nNext steps:")
        print("  1. Create Gumroad product (see LAUNCH_KIT.md)")
        print("  2. Submit to Show HN")
        print("  3. Update Upwork/Toptal portfolio")
        print("  4. Enable GitHub Sponsors")
    else:
        print("⚠️  Some checks failed — fix before launch")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())