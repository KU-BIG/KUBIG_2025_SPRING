from .base import BaseNode
import pymupdf
import os
import re
from .state import GraphState


def get_chunked_output_path(original_filepath: str) -> str:
    """
    원본 파일 경로를 chunking 결과 저장용 경로로 변환
    data/etf_raw/{ticker}/filename.pdf → data/pdf/{ticker}/filename.pdf
    """
    # 파일명에서 ticker 추출
    filename = os.path.basename(original_filepath)
    match = re.match(r"(\d+)_", filename)
    
    if not match:
        # ticker를 찾을 수 없으면 원본 경로 사용
        return original_filepath
        
    ticker = match.group(1)
    filename_without_ext = os.path.splitext(filename)[0]
    
    # data/pdf/{ticker}/filename 구조로 변경
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    chunked_dir = os.path.join(project_root, "data", "pdf", ticker)
    os.makedirs(chunked_dir, exist_ok=True)
    
    return os.path.join(chunked_dir, filename_without_ext)


class SplitPDFFilesNode(BaseNode):

    def __init__(self, batch_size=10, **kwargs):
        super().__init__(**kwargs)
        self.name = "SplitPDFNode"
        self.batch_size = batch_size

    def execute(self, state: GraphState) -> GraphState:
        """
        입력 PDF를 여러 개의 작은 PDF 파일로 분할합니다.

        :param state: GraphState 객체, PDF 파일 경로와 배치 크기 정보를 포함
        :return: 분할된 PDF 파일 경로 목록을 포함한 GraphState 객체
        """
        # PDF 파일 경로와 배치 크기 추출
        filepath = state["filepath"]

        # PDF 파일 열기
        input_pdf = pymupdf.open(filepath)
        num_pages = len(input_pdf)
        print(f"총 페이지 수: {num_pages}")

        ret = []
        # PDF 분할 작업 시작
        for start_page in range(0, num_pages, self.batch_size):
            # 배치의 마지막 페이지 계산 (전체 페이지 수를 초과하지 않도록)
            end_page = min(start_page + self.batch_size, num_pages) - 1

            # chunking 결과용 경로로 변환
            chunked_base_path = get_chunked_output_path(filepath)
            output_file = f"{chunked_base_path}_{start_page:04d}_{end_page:04d}.pdf"
            print(f"분할 PDF 생성: {output_file}")

            # 새로운 PDF 파일 생성 및 페이지 삽입
            with pymupdf.open() as output_pdf:
                output_pdf.insert_pdf(input_pdf, from_page=start_page, to_page=end_page)
                output_pdf.save(output_file)
                ret.append(output_file)

        # 원본 PDF 파일 닫기
        input_pdf.close()

        # 분할된 PDF 파일 경로 목록을 포함한 GraphState 객체 반환
        return GraphState(filepath=filepath, filetype="pdf", split_filepaths=ret)

