from dotenv import load_dotenv
import time
import os
import sys
import io
from pathlib import Path
from src.graphparser.state import GraphState
import src.graphparser.core as parser_core
import src.graphparser.pdf as pdf
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain.schema import Document

# 콘솔 인코딩을 utf-8로 설정
sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(verbose=True)


# 환경 변수 확인을 위한 디버그 출력 추가
print("UPSTAGE_API_KEY:", os.environ.get("UPSTAGE_API_KEY"))
print("환경 변수 로드 위치:", os.getcwd())

# 문서 분할
split_pdf_node = pdf.SplitPDFFilesNode(batch_size=10)

# Layout Analyzer
layout_analyze_node = parser_core.LayoutAnalyzerNode(os.environ.get("UPSTAGE_API_KEY"))

# 페이지 요소 추출
page_element_extractor_node = parser_core.ExtractPageElementsNode()

# 이미지 자르기
image_cropper_node = parser_core.ImageCropperNode()

# 테이블 자르기
table_cropper_node = parser_core.TableCropperNode()

# 페이지별 텍스트 추출
extract_page_text = parser_core.ExtractPageTextNode()

# 페이지별 요약
page_summary_node = parser_core.CreatePageSummaryNode(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# 이미지 요약
image_summary_node = parser_core.CreateImageSummaryNode(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# 테이블 요약
table_summary_node = parser_core.CreateTableSummaryNode(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# 테이블 Markdown 추출
table_markdown_extractor = parser_core.TableMarkdownExtractorNode()

# LangGraph을 생성
workflow = StateGraph(GraphState)

# 노드들을 정의합니다.
workflow.add_node("split_pdf_node", split_pdf_node)
workflow.add_node("layout_analyzer_node", layout_analyze_node)
workflow.add_node("page_element_extractor_node", page_element_extractor_node)
workflow.add_node("image_cropper_node", image_cropper_node)
workflow.add_node("table_cropper_node", table_cropper_node)
workflow.add_node("extract_page_text_node", extract_page_text)
workflow.add_node("page_summary_node", page_summary_node)
workflow.add_node("image_summary_node", image_summary_node)
workflow.add_node("table_summary_node", table_summary_node)
workflow.add_node("table_markdown_node", table_markdown_extractor)

# 각 노드들을 연결합니다.
workflow.add_edge("split_pdf_node", "layout_analyzer_node")
workflow.add_edge("layout_analyzer_node", "page_element_extractor_node")
workflow.add_edge("page_element_extractor_node", "image_cropper_node")
workflow.add_edge("page_element_extractor_node", "table_cropper_node")
workflow.add_edge("page_element_extractor_node", "extract_page_text_node")
workflow.add_edge("image_cropper_node", "page_summary_node")
workflow.add_edge("table_cropper_node", "page_summary_node")
workflow.add_edge("extract_page_text_node", "page_summary_node")
workflow.add_edge("page_summary_node", "image_summary_node")
workflow.add_edge("page_summary_node", "table_summary_node")
workflow.add_edge("image_summary_node", END)
workflow.add_edge("table_summary_node", "table_markdown_node")
workflow.add_edge("table_markdown_node", END)

workflow.set_entry_point("split_pdf_node")

memory = MemorySaver()
graph = workflow.compile()


def process_single_pdf(filepath="data/pdf/20241122_company_22650000.pdf"):
    if not os.path.exists(filepath):
        raise ValueError(f"PDF 파일을 찾을 수 없습니다: {filepath}")

    print(f"처리할 PDF 파일: {filepath}")

    # TypedDict에 맞춰 초기 상태 설정
    initial_state: GraphState = {
        "filepath": filepath,
        "filetype": "pdf",
        "language": "ko",
        "page_numbers": [],
        "batch_size": 10,
        "split_filepaths": [],
        "analyzed_files": [],
        "page_elements": {},
        "page_metadata": {},
        "page_summary": {},
        "images": [],
        "image_summary": {},
        "tables": [],
        "table_summary": {},
        "table_markdown": {},
        "texts": {},
        "text_summary": {},
        "table_summary_data_batches": [],
    }

    try:
        final_state = graph.invoke(initial_state)
        print("PDF 처리가 완료되었습니다.")
        return final_state
    except Exception as e:
        error_message = str(e)
        print(f"PDF 처리 중 오류 발생: {error_message}")


def process_single_xls(filepath="data/etf_raw/ticker/ticker_date_type.xls"):
    """
    XLS 파일 처리 플레이스홀더 함수
    실제 XLS 처리는 src/xls_processor.py와 scripts/process_xls.py에서 담당
    이 함수는 기존 import 호환성을 위해 제공됨
    """
    print(f"XLS 파일 처리는 별도 모듈에서 담당됩니다: {filepath}")
    print("실제 처리를 위해서는 scripts/process_xls.py를 사용하세요.")
    
    # 기본 상태 반환 (실제로는 사용되지 않음)
    return {
        "filepath": filepath,
        "filetype": "xls",
        "status": "redirected_to_xls_processor"
    }