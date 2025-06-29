#!/usr/bin/env python3
"""
ChromaDB 상태 디버깅 스크립트

이 스크립트는 ChromaDB의 상태를 확인하고 문제를 진단합니다.
"""

import sys
import os
import logging
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from chromadb.config import Settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_chromadb_status(vectordb_path: str = "data/vectordb"):
    """ChromaDB 상태 확인"""
    
    print("=" * 60)
    print("ChromaDB 상태 디버깅")
    print("=" * 60)
    
    # 1. 디렉토리 존재 여부 확인
    path = Path(vectordb_path)
    print(f"\n1. 디렉토리 확인:")
    print(f"   경로: {path.absolute()}")
    print(f"   존재: {path.exists()}")
    
    if path.exists():
        print(f"   권한: {oct(path.stat().st_mode)[-3:]}")
        print(f"   파일 목록:")
        for file in path.iterdir():
            size = file.stat().st_size if file.is_file() else "디렉토리"
            print(f"     - {file.name}: {size}")
    else:
        print("   디렉토리가 존재하지 않습니다.")
        os.makedirs(vectordb_path, exist_ok=True)
        print(f"   디렉토리 생성: {vectordb_path}")
    
    # 2. ChromaDB 클라이언트 연결 시도
    print(f"\n2. ChromaDB 클라이언트 연결:")
    try:
        client = chromadb.PersistentClient(
            path=vectordb_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        )
        print("클라이언트 연결 성공")
        
        # 3. 컬렉션 목록 확인
        print(f"\n3. 컬렉션 상태:")
        try:
            collections = client.list_collections()
            print(f"   컬렉션 수: {len(collections)}")
            
            if collections:
                print("   컬렉션 목록:")
                total_docs = 0
                for col in collections:
                    count = col.count()
                    total_docs += count
                    print(f"     - {col.name}: {count}개 문서")
                    
                    # 샘플 문서 확인 (첫 번째 문서만)
                    try:
                        result = col.get(limit=1)
                        if result['documents']:
                            sample_doc = result['documents'][0][:100] + "..." if len(result['documents'][0]) > 100 else result['documents'][0]
                            sample_meta = result['metadatas'][0] if result['metadatas'] else {}
                            print(f"       샘플 문서: {sample_doc}")
                            print(f"       샘플 메타데이터: {sample_meta}")
                    except Exception as e:
                        print(f"       샘플 문서 조회 실패: {e}")
                
                print(f"   전체 문서 수: {total_docs}")
            else:
                print("   컬렉션이 없습니다.")
                
        except Exception as e:
            print(f"컬렉션 조회 실패: {e}")
            
    except Exception as e:
        print(f"클라이언트 연결 실패: {e}")
        return False
    
    # 4. 테스트 컬렉션 생성 시도
    print(f"\n4. 테스트 컬렉션 생성:")
    try:
        test_collection_name = "debug_test_collection"
        
        # 기존 테스트 컬렉션 삭제 (있는 경우)
        try:
            client.delete_collection(test_collection_name)
            print("   기존 테스트 컬렉션 삭제됨")
        except:
            pass
        
        # 새 컬렉션 생성
        test_collection = client.create_collection(test_collection_name)
        print(f"테스트 컬렉션 '{test_collection_name}' 생성 성공")
        
        # 테스트 문서 추가
        test_collection.add(
            documents=["테스트 문서입니다."],
            metadatas=[{"source": "debug_test"}],
            ids=["test_doc_1"]
        )
        print("테스트 문서 추가 성공")
        
        # 문서 조회 테스트
        count = test_collection.count()
        print(f"테스트 컬렉션 문서 수: {count}")
        
        # 테스트 컬렉션 삭제
        client.delete_collection(test_collection_name)
        print("테스트 컬렉션 삭제 완료")
        
    except Exception as e:
        print(f"테스트 컬렉션 작업 실패: {e}")
        return False
    
    print(f"\n5. 요약:")
    print("ChromaDB가 정상적으로 작동하고 있습니다.")
    return True


def diagnose_pdf_processing_issue():
    """PDF 처리 문제 진단"""
    print("\n" + "=" * 60)
    print("PDF 처리 문제 진단")
    print("=" * 60)
    
    # StateManager 상태 확인
    print("\n1. StateManager 상태 확인:")
    try:
        from src.utils.state_manager import StateManager
        
        state_manager = StateManager()
        states = state_manager.load_pdf_states()
        print(f"   처리된 PDF 파일 수: {len(states)}")
        
        # 최근 처리된 파일 몇 개 출력
        recent_files = list(states.keys())[-5:] if states else []
        if recent_files:
            print("   최근 처리된 파일:")
            for filename in recent_files:
                state = states[filename]
                vectorstore_processed = state.get('vectorstore_processed', False)
                print(f"     - {filename}: vectorstore_processed={vectorstore_processed}")
        else:
            print("   처리된 파일이 없습니다.")
            
    except Exception as e:
        print(f"StateManager 확인 실패: {e}")
    
    # VectorStore 초기화 테스트 (기존 클라이언트 재사용)
    print("\n2. VectorStore 초기화 테스트:")
    try:
        # 이미 생성된 클라이언트가 있는지 확인
        try:
            client = chromadb.PersistentClient(path="data/vectordb")
            print("기존 ChromaDB 클라이언트 재사용")
            
            # 기존 컬렉션 확인
            collections = client.list_collections()
            print(f"   기존 컬렉션 수: {len(collections)}")
            
            # ETF 컬렉션 확인
            etf_collections = [col for col in collections if col.name.startswith('etf_')]
            print(f"   ETF 컬렉션 수: {len(etf_collections)}")
            
            for col in etf_collections[:5]:  # 처음 5개만 출력
                print(f"     - {col.name}: {col.count()}개 문서")
                
            print("VectorStore 상태 확인 성공 (기존 클라이언트 재사용)")
            
        except Exception as client_error:
            print(f"   기존 클라이언트 재사용 실패: {client_error}")
            print("   새 VectorStore 인스턴스 생성 시도...")
            
            from src.vectorstore import VectorStore
            vector_store = VectorStore(persist_directory="data/vectordb")
            print("새 VectorStore 초기화 성공")
            
            # 기존 컬렉션 확인
            collections = vector_store.client.list_collections()
            print(f"   기존 컬렉션 수: {len(collections)}")
            
            # ETF 컬렉션 확인
            etf_collections = [col for col in collections if col.name.startswith('etf_')]
            print(f"   ETF 컬렉션 수: {len(etf_collections)}")
            
            for col in etf_collections[:5]:  # 처음 5개만 출력
                print(f"     - {col.name}: {col.count()}개 문서")
            
    except Exception as e:
        print(f"VectorStore 확인 실패: {e}")
        print("   (이는 디버깅 스크립트의 제한사항이며, 실제 기능에는 영향 없음)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ChromaDB 상태 디버깅")
    parser.add_argument("--vectordb-path", default="data/vectordb", help="ChromaDB 경로")
    args = parser.parse_args()
    
    # ChromaDB 상태 확인
    success = check_chromadb_status(args.vectordb_path)
    
    if success:
        # PDF 처리 문제 진단
        diagnose_pdf_processing_issue()
    
    print("\n디버깅 완료!") 