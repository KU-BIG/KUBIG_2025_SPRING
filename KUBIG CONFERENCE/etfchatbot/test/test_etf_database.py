#!/usr/bin/env python3
"""
ETF 데이터베이스 테스트 스크립트

data/etf_database.sqlite 파일의 상태와 내용을 확인합니다.
"""

import sqlite3
import sys
import os
from pathlib import Path

def test_database_connection(db_path: str = "data/etf_database.sqlite"):
    """데이터베이스 연결 테스트"""
    try:
        if not os.path.exists(db_path):
            print(f"데이터베이스 파일이 존재하지 않습니다: {db_path}")
            return False
            
        conn = sqlite3.connect(db_path)
        conn.close()
        print(f"데이터베이스 연결 성공: {db_path}")
        return True
        
    except Exception as e:
        print(f"데이터베이스 연결 실패: {e}")
        return False

def test_table_structure(db_path: str = "data/etf_database.sqlite"):
    """테이블 구조 확인"""
    try:
        conn = sqlite3.connect(db_path)
        
        # 테이블 목록 확인
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        print(f"전체 테이블 수: {len(tables)}")
        print(f"테이블 목록: {tables}")
        
        # 각 테이블의 스키마 확인
        for table in tables:
            print(f"\n테이블 '{table}' 스키마:")
            cursor = conn.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            for col in columns:
                print(f"  - {col[1]} ({col[2]})")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"테이블 구조 확인 실패: {e}")
        return False

def test_data_content(db_path: str = "data/etf_database.sqlite"):
    """데이터 내용 확인"""
    try:
        conn = sqlite3.connect(db_path)
        
        # 가격 데이터 확인
        print("\n가격 데이터 확인:")
        try:
            cursor = conn.execute('SELECT COUNT(*) FROM etf_prices')
            price_count = cursor.fetchone()[0]
            print(f"총 가격 레코드 수: {price_count}")
            
            cursor = conn.execute('SELECT COUNT(DISTINCT ticker) FROM etf_prices')
            unique_tickers = cursor.fetchone()[0]
            print(f"고유 티커 수: {unique_tickers}")
            
            # 최신 가격 데이터 샘플 (date 컬럼 사용)
            cursor = conn.execute('''
                SELECT ticker, date, closing_price, nav 
                FROM etf_prices 
                ORDER BY date DESC 
                LIMIT 5
            ''')
            samples = cursor.fetchall()
            print(f"최신 가격 데이터 샘플:")
            for sample in samples:
                print(f"    {sample[0]}: {sample[1]} - 종가: {sample[2]}, NAV: {sample[3]}")
                
        except Exception as e:
            print(f"가격 데이터 확인 실패: {e}")
        
        # 분배 데이터 확인 (etf_distributions 테이블 사용)
        print("\n분배 데이터 확인:")
        try:
            cursor = conn.execute('SELECT COUNT(*) FROM etf_distributions')
            dist_count = cursor.fetchone()[0]
            print(f"분배 데이터 레코드 수: {dist_count}")
            
            cursor = conn.execute('SELECT COUNT(DISTINCT ticker) FROM etf_distributions')
            unique_dist_tickers = cursor.fetchone()[0]
            print(f"분배 데이터 고유 티커 수: {unique_dist_tickers}")
            
            # 분배 테이블 샘플
            cursor = conn.execute('''
                SELECT ticker, date, stock_name, weight_percent 
                FROM etf_distributions 
                ORDER BY date DESC, weight_percent DESC
                LIMIT 5
            ''')
            dist_samples = cursor.fetchall()
            print(f"분배 데이터 샘플:")
            for sample in dist_samples:
                print(f"    {sample[0]}: {sample[1]} - 종목: {sample[2]}, 비중: {sample[3]}%")
                
        except Exception as e:
            print(f"분배 데이터 확인 실패: {e}")
        
        # 처리 상태 확인
        print("\n처리 상태 확인:")
        try:
            cursor = conn.execute('SELECT COUNT(*) FROM xls_processing_states')
            process_count = cursor.fetchone()[0]
            print(f"처리 상태 레코드 수: {process_count}")
            
            cursor = conn.execute('''
                SELECT file_type, COUNT(*) 
                FROM xls_processing_states 
                GROUP BY file_type
            ''')
            type_counts = cursor.fetchall()
            print(f"파일 타입별 처리 수:")
            for type_count in type_counts:
                print(f"    {type_count[0]}: {type_count[1]}개")
                
        except Exception as e:
            print(f"처리 상태 확인 실패: {e}")
        
        # ETF 다운로드 상태 확인 (새로 추가)
        print("\nETF 다운로드 상태 확인:")
        try:
            cursor = conn.execute('SELECT COUNT(*) FROM etf_download_states')
            download_count = cursor.fetchone()[0]
            print(f"다운로드 상태 레코드 수: {download_count}")
            
            cursor = conn.execute('''
                SELECT doc_type, COUNT(*) 
                FROM etf_download_states 
                GROUP BY doc_type
            ''')
            doc_type_counts = cursor.fetchall()
            print(f"문서 타입별 다운로드 수:")
            for doc_type, count in doc_type_counts:
                print(f"    {doc_type}: {count}개")
                
            # 최신 다운로드 파일 샘플
            cursor = conn.execute('''
                SELECT ticker, filename, doc_type, date 
                FROM etf_download_states 
                ORDER BY downloaded_at DESC 
                LIMIT 5
            ''')
            download_samples = cursor.fetchall()
            print(f"최신 다운로드 파일 샘플:")
            for sample in download_samples:
                print(f"    {sample[0]}: {sample[1]} ({sample[2]}, {sample[3]})")
                
        except Exception as e:
            print(f"ETF 다운로드 상태 확인 실패: {e}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"데이터 내용 확인 실패: {e}")
        return False

def test_file_stats(db_path: str = "data/etf_database.sqlite"):
    """파일 통계 확인"""
    try:
        if os.path.exists(db_path):
            file_size = os.path.getsize(db_path)
            print(f"데이터베이스 파일 크기: {file_size:,} bytes ({file_size/1024:.1f} KB)")
            
            stat = os.stat(db_path)
            import datetime
            mod_time = datetime.datetime.fromtimestamp(stat.st_mtime)
            print(f"마지막 수정 시간: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"파일이 존재하지 않습니다: {db_path}")
            return False
            
        return True
        
    except Exception as e:
        print(f"파일 통계 확인 실패: {e}")
        return False

def main():
    """메인 테스트 실행"""
    print("ETF 데이터베이스 테스트 시작\n")
    
    db_path = "data/etf_database.sqlite"
    
    # 프로젝트 루트로 이동 (test 폴더에서 실행되는 경우)
    current_dir = Path.cwd()
    if current_dir.name == "test":
        os.chdir(current_dir.parent)
        print(f"작업 디렉토리 변경: {Path.cwd()}")
    
    tests = [
        ("파일 통계", test_file_stats),
        ("데이터베이스 연결", test_database_connection),
        ("테이블 구조", test_table_structure),
        ("데이터 내용", test_data_content),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"{test_name} 테스트")
        print(f"{'='*50}")
        
        try:
            if test_func(db_path):
                print(f"{test_name} 테스트 통과")
                passed += 1
            else:
                print(f"{test_name} 테스트 실패")
                failed += 1
        except Exception as e:
            print(f"{test_name} 테스트 중 오류: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"테스트 결과 요약")
    print(f"{'='*50}")
    print(f"통과: {passed}개")
    print(f"실패: {failed}개")
    
    # 실패가 있으면 exit code 1로 종료
    if failed > 0:
        sys.exit(1)
    else:
        print("\n모든 테스트가 성공적으로 완료되었습니다!")
        sys.exit(0)

if __name__ == "__main__":
    main() 