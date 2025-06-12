import os
import glob
import numpy as np
import nibabel as nib
import argparse

def merge_segments_with_label_map(input_dir, output_dir, case_id):
    """
    여러 세그멘테이션 파일을 지정된 라벨 맵에 따라 하나의 파일로 병합합니다.
    
    Args:
        input_dir: 세그멘테이션 파일이 있는 디렉토리 경로
        output_dir: 병합된 결과를 저장할 디렉토리 경로
        case_id: 케이스 ID (예: "case_01")
    """
    # 케이스별 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 라벨 맵 정의
    label_map = {
        1: ["OAR_A_Carotid_L", "OAR_A_Carotid_R"],
        2: ["OAR_Bone_Mandible"],
        3: ["OAR_SpinalCord"],
        4: ["OAR_Glnd_Thyroid"],
        5: ["OAR_Larynx_SG"],
    }
    
    # 참조 이미지 가져오기 (아무 세그멘테이션 파일이나 사용)
    ref_file = None
    for file in glob.glob(os.path.join(input_dir, f"{case_id}_OAR_*.seg.nii.gz")):
        ref_file = file
        break
    
    if not ref_file:
        print(f"오류: {input_dir}에서 세그멘테이션 파일을 찾을 수 없습니다.")
        return
    
    # 참조 이미지 로드
    ref_img = nib.load(ref_file)
    ref_shape = ref_img.shape
    ref_affine = ref_img.affine
    
    # 병합된 세그멘테이션을 저장할 배열 초기화 (배경은 0)
    combined_seg = np.zeros(ref_shape, dtype=np.uint8)
    
    # 각 라벨에 대해 처리
    for label_value, segment_list in label_map.items():
        print(f"라벨 {label_value} 처리 중...")
        
        # 현재 라벨의 세그멘테이션 데이터
        current_label_seg = np.zeros(ref_shape, dtype=np.uint8)
        
        # 각 세그먼트 처리
        for segment in segment_list:
            segment_file = os.path.join(input_dir, f"{case_id}_{segment}.seg.nii.gz")
            
            if not os.path.exists(segment_file):
                print(f"  경고: {segment} 파일을 찾을 수 없습니다 - {segment_file}")
                continue
                
            print(f"  병합 중: {os.path.basename(segment_file)}")
            
            # 세그멘테이션 파일 로드
            seg_img = nib.load(segment_file)
            seg_data = seg_img.get_fdata().astype(np.uint8)
            
            # 현재 세그먼트를 OR 연산으로 병합
            current_label_seg = np.logical_or(current_label_seg, seg_data > 0).astype(np.uint8)
        
        # 현재 라벨의 세그멘테이션을 전체 세그멘테이션에 병합
        # 중복되는 부분은 가장 높은 라벨 값을 가진 것으로 할당
        # (라벨 값이 높을수록 우선순위가 높음)
        combined_seg = np.where(
            (current_label_seg > 0),
            label_value,
            combined_seg
        )
    
    # 환자 번호 추출 (예: "case_01" -> "01")
    patient_num = case_id.split('_')[1]
    
    # 출력 파일 경로
    output_file = os.path.join(output_dir, f"label.nii.gz")
    
    # 병합된 세그멘테이션 저장
    combined_img = nib.Nifti1Image(combined_seg, ref_affine)
    nib.save(combined_img, output_file)
    
    print(f"병합된 세그멘테이션 저장 완료: {output_file}")
    
    return output_file

def merge_all_cases(input_base_dir, output_base_dir, start_case=1, end_case=42):
    """
    모든 케이스에 대해 세그멘테이션 파일을 병합합니다.
    
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
        output_dir = os.path.join(output_base_dir, case_id)
        
        # 입력 디렉토리가 존재하는지 확인
        if not os.path.exists(input_dir):
            print(f"경고: 입력 디렉토리가 존재하지 않습니다 - {input_dir}")
            continue
        
        print(f"\n=== 처리 중: {case_id} ===")
        merge_segments_with_label_map(input_dir, output_dir, case_id)
        print(f"{case_id} 병합 완료\n")
    
    print("모든 케이스 병합 완료!")

def main():
    parser = argparse.ArgumentParser(description='모든 케이스에 대해 세그멘테이션 파일을 라벨 맵에 따라 병합')
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