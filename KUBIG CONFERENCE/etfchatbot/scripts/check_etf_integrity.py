#!/usr/bin/env python3
"""
ETF 파일 다운로드 상태 및 무결성 확인 스크립트

etf_download_states 테이블을 기반으로 모든 파일 타입 (price, distribution, agreement, prospectus, monthly)의
다운로드 상태와 무결성을 확인합니다.
"""

import os
import sys
import argparse
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.state_manager import StateManager


def print_separator(title: str):
    """구분선과 제목 출력"""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")


def check_overall_integrity():
    """전체 ETF 파일 무결성 확인"""
    print_separator("ETF 파일 무결성 전체 확인")
    
    state_manager = StateManager()
    integrity_data = state_manager.check_etf_file_integrity()
    
    if not integrity_data:
        print("무결성 데이터를 가져올 수 없습니다.")
        return
    
    complete_tickers = []
    incomplete_tickers = []
    
    for ticker, data in integrity_data.items():
        if data["complete"]:
            complete_tickers.append(ticker)
        else:
            incomplete_tickers.append(ticker)
            
        print(f"\n{ticker}:")
        print(f"다운로드된 타입: {', '.join(sorted(data['downloaded_types']))}")
        
        if data["missing_types"]:
            print(f"누락된 타입: {', '.join(sorted(data['missing_types']))}")
        
        print(f"완성도: {data['completion_rate']:.1f}%")
    
    print(f"\n요약:")
    print(f"  완전한 ETF: {len(complete_tickers)}개")
    print(f"  불완전한 ETF: {len(incomplete_tickers)}개")
    
    if complete_tickers:
        print(f"  완전한 ETF 목록: {', '.join(complete_tickers)}")
    
    if incomplete_tickers:
        print(f"  불완전한 ETF 목록: {', '.join(incomplete_tickers)}")


def check_ticker_integrity(ticker: str):
    """특정 티커의 상세 무결성 확인"""
    print_separator(f"ETF {ticker} 상세 무결성 확인")
    
    state_manager = StateManager()
    integrity_data = state_manager.check_etf_file_integrity(ticker=ticker)
    
    if not integrity_data:
        print(f"{ticker}에 대한 데이터를 찾을 수 없습니다.")
        return
    
    print(f"ETF: {ticker}")
    print(f"완전한 파일 세트: {'예' if integrity_data['complete'] else '아니오'}")
    
    print(f"\n파일 타입별 상세 정보:")
    for doc_type, details in integrity_data["details"].items():
        status_icon = "O" if details["status"] == "downloaded" else "X"
        print(f"  {status_icon} {doc_type.upper():<12}: ", end="")
        
        if details["status"] == "downloaded":
            print(f"{details['count']}개 파일, 최신: {details['latest_date']}")
        else:
            print("누락됨")
    
    if integrity_data["downloaded_types"]:
        print(f"\n다운로드된 타입: {', '.join(sorted(integrity_data['downloaded_types']))}")
    
    if integrity_data["missing_types"]:
        print(f"누락된 타입: {', '.join(sorted(integrity_data['missing_types']))}")


def check_download_stats():
    """다운로드 통계 확인"""
    print_separator("ETF 다운로드 통계")
    
    state_manager = StateManager()
    stats = state_manager.get_etf_download_stats()
    
    if not stats:
        print("통계 데이터를 가져올 수 없습니다.")
        return
    
    print(f"전체 통계:")
    print(f"    총 다운로드 파일: {stats['total_files']}개")
    print(f"    총 파일 크기: {stats['total_size_bytes']:,} bytes")
    print(f"    최근 24시간 다운로드: {stats['recent_downloads_24h']}개")
    
    print(f"\n문서 타입별 통계:")
    for doc_type in stats['expected_doc_types']:
        if doc_type in stats['by_doc_type_detailed']:
            detail = stats['by_doc_type_detailed'][doc_type]
            print(f"  {doc_type.upper():<12}: {detail['count']}개 파일")
            print(f"                ({detail['total_size_bytes']:,} bytes)")
            print(f"                최초: {detail['first_download']}")
            print(f"                최근: {detail['last_download']}")
        else:
            print(f"  {doc_type.upper():<12}: 0개 파일")
    
    print(f"\n티커별 통계:")
    for ticker, detail in stats['by_ticker_detailed'].items():
        print(f"  {ticker}: {detail['total_files']}개 파일")
        print(f"         고유 타입: {detail['unique_doc_types']}개")
        print(f"         완성도: {detail['completion_rate']:.1f}%")
        print(f"         최종 다운로드: {detail['last_download']}")


def check_file_type_details(doc_type: str):
    """특정 파일 타입의 상세 정보 확인"""
    print_separator(f"{doc_type.upper()} 파일 상세 정보")
    
    state_manager = StateManager()
    files_data = state_manager.get_downloaded_files_by_type(doc_type=doc_type)
    
    if not files_data or files_data.get('total_count', 0) == 0:
        print(f"{doc_type.upper()} 타입의 다운로드된 파일이 없습니다.")
        return
    
    print(f"{doc_type.upper()} 파일 총 {files_data['total_count']}개")
    print(f"\n파일 목록:")
    
    for i, file_info in enumerate(files_data['files'][:10], 1):  # 최신 10개만 표시
        print(f"  {i:2d}. {file_info['filename']}")
        print(f"      티커: {file_info['ticker']}")
        print(f"      날짜: {file_info['date']}")
        print(f"      다운로드: {file_info['downloaded_at']}")
        print(f"      크기: {file_info['file_size']:,} bytes" if file_info['file_size'] else "      크기: 알 수 없음")
        print()
    
    if files_data['total_count'] > 10:
        print(f"  ... 및 {files_data['total_count'] - 10}개 추가 파일")


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="ETF 파일 다운로드 상태 및 무결성 확인")
    parser.add_argument(
        "--ticker", 
        type=str, 
        help="특정 티커의 상세 무결성 확인"
    )
    parser.add_argument(
        "--file-type", 
        type=str, 
        choices=["price", "distribution", "agreement", "prospectus", "monthly"],
        help="특정 파일 타입의 상세 정보 확인"
    )
    parser.add_argument(
        "--stats-only", 
        action="store_true",
        help="통계만 표시"
    )
    parser.add_argument(
        "--integrity-only", 
        action="store_true",
        help="무결성 확인만 수행"
    )
    
    args = parser.parse_args()
    
    # 현재 디렉토리를 프로젝트 루트로 변경
    if Path.cwd().name in ["scripts", "test"]:
        os.chdir(Path.cwd().parent)
    
    print("ETF 파일 다운로드 상태 확인 시작")
    print(f"데이터베이스 경로: {Path('data/etf_database.sqlite').absolute()}")
    
    # 데이터베이스 파일 존재 확인
    if not Path("data/etf_database.sqlite").exists():
        print("etf_database.sqlite 파일이 존재하지 않습니다.")
        print("   먼저 ETF 데이터를 다운로드하고 처리하세요.")
        return
    
    try:
        if args.ticker:
            check_ticker_integrity(args.ticker)
        elif args.file_type:
            check_file_type_details(args.file_type)
        elif args.stats_only:
            check_download_stats()
        elif args.integrity_only:
            check_overall_integrity()
        else:
            # 전체 확인
            check_download_stats()
            check_overall_integrity()
    
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nETF 파일 상태 확인 완료")


if __name__ == "__main__":
    main() 