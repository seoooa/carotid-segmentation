import os
import glob
import numpy as np
import nibabel as nib
import argparse

def merge_carotid_segments(input_dir, output_base_dir, case_id):
    """
    좌우 경동맥(Carotid L, R) 세그멘테이션을 하나의 파일로 병합합니다.
    세그먼트 영역은 1, 아닌 부분은 0으로 표시합니다.
    
    Args:
        input_dir: 세그멘테이션 파일이 있는 디렉토리 경로
        output_base_dir: 병합된 결과를 저장할 기본 디렉토리 경로
        case_id: 케이스 ID (예: "case_01")
    """
    # 환자 번호 추출 (예: "case_01" -> "01")
    patient_num = case_id.split('_')[1]
    
    # 케이스별 출력 디렉토리 생성
    output_dir = os.path.join(output_base_dir, case_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # 좌우 경동맥 파일 경로
    carotid_l_file = os.path.join(input_dir, f"{case_id}_OAR_A_Carotid_L.seg.nii.gz")
    carotid_r_file = os.path.join(input_dir, f"{case_id}_OAR_A_Carotid_R.seg.nii.gz")
    
    # 파일 존재 여부 확인
    if not os.path.exists(carotid_l_file):
        print(f"오류: 좌측 경동맥 파일을 찾을 수 없습니다 - {carotid_l_file}")
        return
    
    if not os.path.exists(carotid_r_file):
        print(f"오류: 우측 경동맥 파일을 찾을 수 없습니다 - {carotid_r_file}")
        return
    
    # 세그멘테이션 파일 로드
    carotid_l_img = nib.load(carotid_l_file)
    carotid_l_data = carotid_l_img.get_fdata().astype(np.uint8)
    
    carotid_r_img = nib.load(carotid_r_file)
    carotid_r_data = carotid_r_img.get_fdata().astype(np.uint8)
    
    # 좌우 경동맥 병합 (OR 연산)
    # 두 세그멘테이션 중 하나라도 1이면 결과도 1
    combined_data = np.logical_or(carotid_l_data > 0, carotid_r_data > 0).astype(np.uint8)
    
    # 출력 파일 경로
    output_file = os.path.join(output_dir, f"carotid.nii.gz")
    
    # 병합된 세그멘테이션 저장
    combined_img = nib.Nifti1Image(combined_data, carotid_l_img.affine)
    nib.save(combined_img, output_file)
    
    print(f"병합된 경동맥 세그멘테이션 저장 완료: {output_file}")
    
    return output_file

def merge_all_cases(input_base_dir, output_base_dir, start_case=1, end_case=42):
    """
    모든 케이스에 대해 좌우 경동맥 세그멘테이션을 병합합니다.
    
    Args:
        input_base_dir: 세그멘테이션 파일이 있는 기본 디렉토리 경로
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
        
        # 입력 디렉토리가 존재하는지 확인
        if not os.path.exists(input_dir):
            print(f"경고: 입력 디렉토리가 존재하지 않습니다 - {input_dir}")
            continue
        
        print(f"\n=== 처리 중: {case_id} ===")
        merge_carotid_segments(input_dir, output_base_dir, case_id)
        print(f"{case_id} 병합 완료\n")
    
    print("모든 케이스 병합 완료!")

def main():
    parser = argparse.ArgumentParser(description='모든 케이스에 대해 좌우 경동맥 세그멘테이션을 병합')
    parser.add_argument('--input_base_dir', type=str, required=True, 
                        help='세그멘테이션 파일이 있는 기본 디렉토리 경로 (예: "data/HaN_Seg")')
    parser.add_argument('--output_base_dir', type=str, required=True, 
                        help='병합된 결과를 저장할 기본 디렉토리 경로 (예: "data/Carotid_Merged")')
    parser.add_argument('--start_case', type=int, default=1, 
                        help='시작 케이스 번호 (기본값: 1)')
    parser.add_argument('--end_case', type=int, default=42, 
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