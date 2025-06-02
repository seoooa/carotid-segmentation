import os
import glob
import numpy as np
import nibabel as nib
import argparse

def merge_segmentations(input_dir, output_file):
    """
    모든 segmentation 파일을 하나의 파일로 병합합니다.
    각 파일에는 다른 레이블 값이 할당됩니다.
    
    Args:
        input_dir: segmentation 파일이 있는 디렉토리 경로
        output_file: 병합된 결과를 저장할 파일 경로
    """
    # 입력 디렉토리에서 모든 segmentation 파일 찾기
    seg_files = glob.glob(os.path.join(input_dir, "*OAR_*.seg.nii.gz"))
    
    if not seg_files:
        print(f"오류: {input_dir}에서 segmentation 파일을 찾을 수 없습니다.")
        return
    
    print(f"총 {len(seg_files)}개의 segmentation 파일을 찾았습니다.")
    
    # 첫 번째 파일을 참조 이미지로 사용
    ref_img = nib.load(seg_files[0])
    ref_shape = ref_img.shape
    ref_affine = ref_img.affine
    
    # 병합된 segmentation을 저장할 배열 초기화 (배경은 0)
    combined_seg = np.zeros(ref_shape, dtype=np.uint8)
    
    # 각 segmentation 파일에 대해 레이블 값 할당
    for label_value, seg_file in enumerate(sorted(seg_files), start=1):
        file_name = os.path.basename(seg_file)
        
        print(f"병합 중: {file_name} (레이블 값: {label_value})")
        
        # Segmentation 파일 로드
        seg_img = nib.load(seg_file)
        seg_data = seg_img.get_fdata().astype(np.uint8)
        
        # 레이블 값 할당
        # 현재 segmentation이 1인 위치에 레이블 값을 할당
        # 이미 할당된 영역이 있으면 우선순위에 따라 덮어씀 (여기서는 나중에 처리된 파일이 더 높은 우선순위)
        combined_seg = np.where(
            (seg_data > 0),
            label_value,
            combined_seg
        )
    
    # 최종 병합된 segmentation 저장
    combined_img = nib.Nifti1Image(combined_seg, ref_affine)
    nib.save(combined_img, output_file)
    print(f"병합된 segmentation 저장 완료: {output_file}")
    
    return output_file

def merge_all_cases(input_base_dir, output_base_dir, start_case=1, end_case=42):
    """
    모든 케이스에 대해 segmentation 파일을 병합합니다.
    
    Args:
        input_base_dir: segmentation 파일이 있는 기본 디렉토리 경로
        output_base_dir: 병합된 결과를 저장할 기본 디렉토리 경로
        start_case: 시작 케이스 번호
        end_case: 종료 케이스 번호
    """
    # 출력 기본 디렉토리가 없으면 생성
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 각 케이스 처리
    for case_num in range(start_case, end_case + 1):
        case_id = f"case_{case_num:02d}"  # 예: case_01, case_02, ...
        
        input_dir = os.path.join(input_base_dir, case_id)
        output_dir = os.path.join(output_base_dir, case_id)
        
        # 입력 디렉토리가 존재하는지 확인
        if not os.path.exists(input_dir):
            print(f"경고: 입력 디렉토리가 존재하지 않습니다 - {input_dir}")
            continue
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 병합된 파일 경로
        output_file = os.path.join(output_dir, f"{case_id}_all_segments_combined.nii.gz")
        
        print(f"\n=== 처리 중: {case_id} ===")
        merge_segmentations(input_dir, output_file)
        print(f"{case_id} 병합 완료\n")
    
    print("모든 케이스 병합 완료!")

def main():
    parser = argparse.ArgumentParser(description='모든 segmentation 파일을 하나의 파일로 병합')
    parser.add_argument('--input_base_dir', type=str, required=True, 
                        help='segmentation 파일이 있는 기본 디렉토리 경로 (예: "data/HaN_Seg_nifti")')
    parser.add_argument('--output_base_dir', type=str, required=True, 
                        help='병합된 결과를 저장할 기본 디렉토리 경로 (예: "data/HaN_Seg_merged")')
    parser.add_argument('--start_case', type=int, default=1, 
                        help='시작 케이스 번호 (기본값: 1)')
    parser.add_argument('--end_case', type=int, default=1, 
                        help='종료 케이스 번호 (기본값: 42)')
    
    args = parser.parse_args()
    
    merge_all_cases(
        args.input_base_dir, 
        args.output_base_dir, 
        args.start_case, 
        args.end_case
    )

if __name__ == "__main__":
    main()