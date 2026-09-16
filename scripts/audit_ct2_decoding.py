from __future__ import annotations

from benchmarking.smoke_latency import build_opennmt_engine
from reparos.serving.ctranslate2 import CTranslate2Predictor


CHECKPOINT = "artifacts/opennmt-production-kaggle-t4-v1/reparos_base_step_25000.pt"
TOKENIZER = "artifacts/tokenizer-production-v1/tokenizer.model"
DECODING = "artifacts/opennmt-production-kaggle-t4-v1/decoding-config.json"
CT2_MODEL = "artifacts/opennmt-production-kaggle-t4-v1-ctranslate2-float32"


def main() -> None:
    query = "ho guom"
    engine = build_opennmt_engine(CHECKPOINT, TOKENIZER, DECODING)
    predictor = CTranslate2Predictor(CT2_MODEL, device="cpu", compute_type="float32")
    try:
        scores, hypotheses = engine.infer_list([query])
        print("OPENNMT")
        for score, hypothesis in zip(scores[0], hypotheses[0], strict=True):
            print(float(score), hypothesis)
        pieces = predictor.tokenizer.processor.encode(query, out_type=str)
        for beam in (10, 20, 50):
            for patience in (1.0, 2.0, 5.0):
                for penalty in (0.0, 0.6, 1.0, 1.2):
                    result = predictor.translator.translate_batch(
                        [pieces], beam_size=beam, patience=patience,
                        num_hypotheses=min(10, beam), length_penalty=penalty,
                        min_decoding_length=0, max_decoding_length=100,
                        return_scores=True,
                    )[0]
                    decoded = [predictor.tokenizer.processor.decode(tokens) for tokens in result.hypotheses]
                    rank = decoded.index("hồ gươm") + 1 if "hồ gươm" in decoded else 0
                    print(f"beam={beam} patience={patience} lp={penalty} rank={rank} top1={decoded[0]!r}")
    finally:
        engine.terminate()


if __name__ == "__main__":
    main()
