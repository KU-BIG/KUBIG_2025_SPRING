"""
전체 테스트 실행 스크립트

모든 테스트를 순차적으로 실행하고 결과를 요약합니다.
"""

import sys
import subprocess
import os
from pathlib import Path

def run_test_script(script_name: str, description: str):
    """개별 테스트 스크립트 실행"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"실행: {script_name}")
    print(f"{'='*60}")
    
    try:
        # 현재 디렉토리에서 테스트 스크립트 실행
        result = subprocess.run([
            sys.executable, 
            f"test/{script_name}"
        ], capture_output=True, text=True, timeout=120)
        
        # 표준 출력 출력
        if result.stdout:
            print(result.stdout)
        
        # 표준 에러 출력
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # 반환 코드 확인
        if result.returncode == 0:
            print(f"{description} 성공")
            return True
        else:
            print(f"{description} 실패 (exit code: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"{description} 타임아웃 (120초 초과)")
        return False
    except FileNotFoundError:
        print(f"테스트 스크립트를 찾을 수 없습니다: test/{script_name}")
        return False
    except Exception as e:
        print(f"{description} 실행 중 오류: {e}")
        return False

def check_prerequisites():
    """테스트 실행 전 필수 조건 확인"""
    print("테스트 실행 전 필수 조건 확인")
    print("="*50)
    
    checks = []
    
    # 1. Python 버전 확인
    python_version = sys.version_info
    if python_version >= (3, 7):
        print(f"Python 버전: {python_version.major}.{python_version.minor}.{python_version.micro}")
        checks.append(True)
    else:
        print(f"Python 버전이 너무 낮습니다: {python_version.major}.{python_version.minor}")
        checks.append(False)
    
    # 2. 작업 디렉토리 확인
    current_dir = Path.cwd()
    if current_dir.name == "test":
        os.chdir(current_dir.parent)
        print(f"작업 디렉토리 변경: {Path.cwd()}")
    
    expected_files = ["Jenkinsfile", "requirements.txt", "README.md"]
    for file in expected_files:
        if os.path.exists(file):
            print(f"프로젝트 파일 확인: {file}")
            checks.append(True)
        else:
            print(f"프로젝트 파일 없음: {file}")
            checks.append(False)
    
    # 3. 필수 디렉토리 확인
    required_dirs = ["data", "src", "scripts", "test"]
    for dir_name in required_dirs:
        if os.path.exists(dir_name) and os.path.isdir(dir_name):
            print(f"필수 디렉토리 확인: {dir_name}/")
            checks.append(True)
        else:
            print(f"필수 디렉토리 없음: {dir_name}/")
            checks.append(False)
    
    return all(checks)

def main():
    """메인 테스트 실행"""
    print("전체 ETF 데이터 파이프라인 테스트 시작")
    print("="*60)
    
    # 필수 조건 확인
    if not check_prerequisites():
        print("\n필수 조건을 만족하지 않아 테스트를 중단합니다.")
        sys.exit(1)
    
    # 실행할 테스트 목록
    tests = [
        ("test_etf_database.py", "ETF 데이터베이스 테스트"),
    ]
    
    results = []
    total_tests = len(tests)
    passed_tests = 0
    
    # 각 테스트 실행
    for script_name, description in tests:
        success = run_test_script(script_name, description)
        results.append((description, success))
        if success:
            passed_tests += 1
    
    # 최종 결과 요약
    print(f"\n{'='*60}")
    print(f"전체 테스트 결과 요약")
    print(f"{'='*60}")
    
    for description, success in results:
        status = "통과" if success else "실패"
        print(f"{status}: {description}")
    
    print(f"\n테스트 통계:")
    print(f"  총 테스트: {total_tests}개")
    print(f"  통과: {passed_tests}개")
    print(f"  실패: {total_tests - passed_tests}개")
    print(f"  성공률: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print(f"\n모든 테스트가 성공적으로 완료되었습니다!")
        sys.exit(0)
    else:
        print(f"\n{total_tests - passed_tests}개의 테스트가 실패했습니다.")
        sys.exit(1)

if __name__ == "__main__":
    main() 