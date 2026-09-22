from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CASES: dict[str, list[tuple[str, str]]] = {
    "clean_protected": [
        ("Q1 Tower", "Q1 Tower"),
        ("BV Bank", "BV Bank"),
        ("VinFast", "VinFast"),
        ("The Manor Central Park", "The Manor Central Park"),
        ("Starbucks Reserve", "Starbucks Reserve"),
        ("McDonald's Nguyễn Huệ", "McDonald's Nguyễn Huệ"),
        ("đường D2", "đường D2"),
        ("quốc lộ 1A", "quốc lộ 1A"),
    ],
    "abbreviation": [
        ("bv bạch mai", "bệnh viện bạch mai"),
        ("tp hcm", "thành phố hồ chí minh"),
        ("q1 tp hcm", "quận 1 thành phố hồ chí minh"),
        ("đ nguyễn trãi q1", "đường nguyễn trãi quận 1"),
        ("đh bách khoa hn", "đại học bách khoa hà nội"),
        ("ubnd p bến nghé q1", "ủy ban nhân dân phường bến nghé quận 1"),
        ("kcn tân bình", "khu công nghiệp tân bình"),
        ("thpt chuyên lê hồng phong", "trung học phổ thông chuyên lê hồng phong"),
    ],
    "location_acronym": [
        ("ubnd q1", "ủy ban nhân dân quận 1"),
        ("bv chợ rẫy", "bệnh viện chợ rẫy"),
        ("kcn sóng thần", "khu công nghiệp sóng thần"),
        ("đh quốc gia hn", "đại học quốc gia hà nội"),
        ("thpt trần đại nghĩa", "trung học phổ thông trần đại nghĩa"),
        ("tt bến lức", "thị trấn bến lức"),
        ("tx bến cát", "thị xã bến cát"),
        ("h đan phượng", "huyện đan phượng"),
    ],
    "abbreviation_boundary": [
        ("bvbạchmai", "bệnh viện bạch mai"),
        ("đhbáchkhoa hn", "đại học bách khoa hà nội"),
        ("ubndp bếnnghé q1", "ủy ban nhân dân phường bến nghé quận 1"),
        ("kcântanbinh", "khu công nghiệp tân bình"),
        ("đnguyentrai q1", "đường nguyễn trãi quận 1"),
        ("bv bm qdd hn", "bệnh viện bạch mai quận đống đa hà nội"),
        ("thptclhp", "trung học phổ thông chuyên lê hồng phong"),
        ("tp.hcm q1", "thành phố hồ chí minh quận 1"),
    ],
    "diacritics_boundary": [
        ("sanbay noibai", "sân bay nội bài"),
        ("benhvien bachmai", "bệnh viện bạch mai"),
        ("caugiay hanoi", "cầu giấy hà nội"),
        ("thanhpho hochiminh", "thành phố hồ chí minh"),
        ("ngatu so", "ngã tư sở"),
        ("dienbien phu", "điện biên phủ"),
        ("hamchui nguyenhuucanh", "hầm chui nguyễn hữu cảnh"),
        ("truongdaihoc bachkhoa", "trường đại học bách khoa"),
    ],
    "telex_malformed": [
        ("ddiaj chij", "địa chỉ"),
        ("phuwowngf", "phường"),
        ("thanhf phoos", "thành phố"),
        ("bexnh vieejn", "bệnh viện"),
        ("dduwowngf", "đường"),
        ("qujaanj", "quận"),
        ("thij xax", "thị xã"),
        ("khu coong nghieepj", "khu công nghiệp"),
    ],
    "vni_and_input_method": [
        ("tru7o7ng2", "trường"),
        ("be65nh5 vie65n6", "bệnh viện"),
        ("d9u7o7ng2", "đường"),
        ("phu7o7ng2", "phường"),
        ("quan65", "quận"),
        ("nga4 tu7", "ngã tư"),
        ("thanh2 pho61", "thành phố"),
        ("be61n nghe71", "bến nghé"),
    ],
    "lexical_typo": [
        ("ham chui nguyen hune canh", "hầm chui nguyễn hữu cảnh"),
        ("benh vien chor ray", "bệnh viện chợ rẫy"),
        ("linmart", "winmart"),
        ("toa nha bitexto", "tòa nhà bitexco"),
        ("duong hoang dieu2", "đường hoàng diệu"),
        ("cho ben thanhg", "chợ bến thành"),
        ("nga tu bay hienn", "ngã tư bảy hiền"),
        ("cau vuot cong hoaf", "cầu vượt cộng hòa"),
    ],
    "number_symbol": [
        ("12 5 nguyễn trãi q1", "12/5 nguyễn trãi quận 1"),
        ("21 15 lý thường kiệt", "21/15 lý thường kiệt"),
        ("195 9c nguyễn trãi", "195/9c nguyễn trãi"),
        ("hẻm 42 21 đường số 5", "hẻm 42/21 đường số 5"),
        ("quốc lộ 1 a", "quốc lộ 1A"),
        ("đường d 2", "đường D2"),
        ("15a 3 lê lợi", "15A/3 lê lợi"),
        ("km 12 500 quốc lộ 1a", "km 12+500 quốc lộ 1A"),
    ],
    "short_ambiguous": [
        ("ho guom", "hồ gươm"),
        ("nga tu so", "ngã tư sở"),
        ("noi bai", "nội bài"),
        ("bach mai", "bạch mai"),
        ("mai dich", "mai dịch"),
        ("cat linh", "cát linh"),
        ("lang ha", "láng hạ"),
        ("yen nghia", "yên nghĩa"),
    ],
    "mixed_long": [
        ("15 dduwowngf lys thuowngf kieetj q1 tphcm", "15 đường lý thường kiệt quận 1 thành phố hồ chí minh"),
        ("bv bachmai q dongda hanoi", "bệnh viện bạch mai quận đống đa hà nội"),
        ("sanbay noij bai socson hanoi", "sân bay nội bài sóc sơn hà nội"),
        ("ubnd p bennghe q1 tphcm", "ủy ban nhân dân phường bến nghé quận 1 thành phố hồ chí minh"),
        ("truong dh bachkhoa thanhpho hochiminh", "trường đại học bách khoa thành phố hồ chí minh"),
        ("hem 12 5 nguyen trai q1", "hẻm 12/5 nguyễn trãi quận 1"),
        ("kcn tanbinh d truongchinh", "khu công nghiệp tân bình đường trường chinh"),
        ("nga4tu7so73 d9o61ng d9a ha2no65i5", "ngã tư sở đống đa hà nội"),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Write reviewed user-centric ReparoS diagnostic cases")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for error_type, cases in CASES.items():
        for index, (source, target) in enumerate(cases):
            digest = hashlib.sha256(f"{error_type}\0{source}\0{target}".encode()).hexdigest()[:16]
            rows.append({
                "query_id": f"user:{error_type}:{index:02d}:{digest}",
                "input": source,
                "expected": target,
                "error_type": error_type,
                "split": "user-centric-diagnostic",
                "review_status": "manually_specified_v1",
            })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "schema_version": 2,
        "purpose": "frozen user-centric diagnostic; never use for training",
        "review_status": "assistant-reviewed; requires domain-owner sign-off",
        "strict_exact": "Unicode NFC and whitespace normalized; case and punctuation preserved",
        "search_equivalent_note": "report separately after an explicit domain-approved normalization policy",
        "rows": len(rows),
        "buckets": {key: len(value) for key, value in CASES.items()},
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
