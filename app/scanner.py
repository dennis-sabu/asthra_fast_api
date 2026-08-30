from paddleocr import PaddleOCRVL


class IDScanner:
    def __init__(self):
        print("========================================")
        print("Loading PaddleOCR-VL 1.6...")
        print("========================================")

        self.pipeline = PaddleOCRVL(
            pipeline_version="v1.6"
        )

        print("PaddleOCR-VL loaded successfully!")
        print("========================================")

    def scan(self, image_path: str):
        """
        Run PaddleOCR-VL on an ID card image.
        Returns the raw OCR result.
        """

        output = self.pipeline.predict(image_path)

        results = []

        for result in output:
            data = result.json

            if callable(data):
                data = data()

            results.append(data)

        return results

    def extract_text_blocks(self, raw_results):
        """
        Extract text blocks from PaddleOCR-VL output.
        """

        text_blocks = []

        for result in raw_results:
            res = result.get("res", result)

            parsing_list = res.get("parsing_res_list", [])

            for block in parsing_list:
                if block.get("block_label") == "text":
                    text = block.get("block_content", "").strip()

                    if text:
                        text_blocks.append({
                            "text": text,
                            "bbox": block.get("block_bbox")
                        })

                elif block.get("block_label") == "paragraph_title":
                    text = block.get("block_content", "").strip()

                    if text:
                        text_blocks.append({
                            "text": text,
                            "bbox": block.get("block_bbox")
                        })

        return text_blocks

    def extract_name(self, text_blocks):
        """
        Try to identify the person's name from the ID-card text.

        This is intentionally conservative:
        we reject obvious college/header/ID/degree text
        and then choose the strongest remaining candidate.
        """

        candidates = []

        reject_words = [
            "ST.JOSEPH",
            "ST. JOSEPH",
            "COLLEGE",
            "ENGINEERING",
            "TECHNOLOGY",
            "AUTONOMOUS",
            "B.TECH",
            "BTECH",
            "ECS",
            "ECE",
            "COMPUTER",
            "PRINCIPAL",
            "SIGNATURE",
            "MANAGED",
            "DIOCESE",
        ]

        for block in text_blocks:
            text = block["text"].strip()

            upper = text.upper()

            # Reject obvious non-name text.
            if any(word in upper for word in reject_words):
                continue

            # Reject strings containing numbers.
            if any(char.isdigit() for char in text):
                continue

            # Reject very short text.
            if len(text) < 4:
                continue

            # Names normally contain alphabetic characters and spaces.
            if not all(
                char.isalpha() or char in " .'-"
                for char in text
            ):
                continue

            words = text.split()

            # A person's name usually has 2+ words.
            if len(words) < 2:
                continue

            # Reject extremely long text blocks.
            if len(words) > 5:
                continue

            # Score the candidate.
            score = 0

            # Two or three words are ideal.
            if 2 <= len(words) <= 3:
                score += 10

            # Reasonable name length.
            if 8 <= len(text) <= 40:
                score += 5

            # Capitalized words look like a person's name.
            if all(
                word[0].isupper()
                for word in words
                if word
            ):
                score += 5

            candidates.append({
                "text": text,
                "score": score,
                "bbox": block.get("bbox")
            })

        if not candidates:
            return None

        # Highest-scoring candidate.
        candidates.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return candidates[0]["text"]