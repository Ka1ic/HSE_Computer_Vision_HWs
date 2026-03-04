from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def find_vertical_seam(
    image: np.ndarray,
) -> np.ndarray:
    # 0. Переводим изображение в grayscale: энергия и переходы считаем по яркости.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape

    # m[i, j] — минимальная накопленная стоимость шва,
    # который заканчивается в пикселе (i, j).
    m = np.zeros((h, w), dtype=np.float32)
    # backtrack[i, j] хранит смещение по столбцу для предыдущей строки:
    # -1 (слева), 0 (сверху), +1 (справа).
    backtrack = np.zeros((h, w), dtype=np.int16)

    # Градиенты Sobel по x и y.
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # 1. База динамики для первой строки: локальная энергия без переходов.
    m[0] = np.abs(dx[0]) + np.abs(dy[0])
    # Шов начинается заполняться после полного подсчета m
    # Для востановления путей используйте backtrack
    seam = np.zeros(h, dtype=np.int32)

    for i in range(1, h):
        # Реализуйте динамику
        c = np.full((3, w), np.inf)
        c[1, 1:-1] = np.abs(gray[i, 2:] - gray[i, 0:-2]) # C_u
        c[1, 0] = np.abs(gray[i, 1] - gray[i, 0]) # по краям чтоб можно было идти вниз
        c[1, -1] = np.abs(gray[i, -1] - gray[i, -2])

        c[0, 1:] = np.abs(gray[i - 1, 1:] - gray[i, 0:-1]) # C_l
        c[0] += c[1]

        c[2, :-1] = np.abs(gray[i - 1, :-1] - gray[i, 1:]) # C_r
        c[2] += c[1]

        c[0, 1:] += m[i - 1, :-1] # m[i - 1, j - 1] + c_l
        c[1] += m[i - 1] # m[i - 1, j] + c_u
        c[2, :-1] += m[i - 1, 1:] # m[i - 1, j + 1] + c_r]

        m[i] = np.min(c, axis=0)
        backtrack[i] = np.argmin(c, axis=0) - 1
  
    # Реализуйте восстановление шва
    seam[h - 1] = np.argmin(m[h-1])
    for i in range(h - 2, -1, -1):
        seam[i] = seam[i + 1] + backtrack[i + 1, seam[i + 1]]

    return seam


def remove_vertical_seam(image: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """удалить 1 вертикальный шов

    Args:
        image (np.ndarray): изображение (h, w, c)
        seam (np.ndarray): массив размера h. В iй позиции стоит индекс j
          -> шов проходит через (i,j)

    Returns:
        np.ndarray: изображение размера (h, w - 1, c)
    """
    # Hint: советуем использовать булевые маски
    h, w, c = image.shape
    mask = np.ones((h, w), dtype=bool)
    mask[np.arange(h), seam] = False

    return image[mask].reshape(h, w - 1, c)


def carve_vertical(
    image: np.ndarray,
    num_seams: int,
) -> np.ndarray:
    assert num_seams > 0 and num_seams < image.shape[1]

    iterator = tqdm(range(num_seams), desc="Removing columns", unit="seam")

    out = image
    for _ in iterator:
        seam = find_vertical_seam(out)
        out = remove_vertical_seam(out, seam)
    return out


def carve_horizontal(
    image: np.ndarray,
    num_seams: int,
) -> np.ndarray:
    assert num_seams > 0 and num_seams < image.shape[0]

    # Просто поверните, используйте функцию для удаления вертикальных швов
    # и поверните обратно
    image = image.transpose(1, 0, 2) # транспонирую
    out = carve_vertical(image, num_seams)
    out = out.transpose(1, 0, 2)

    return out


def main() -> None:
    ASSETS_DIR = Path(__file__).parent.resolve() / "assets" / "seam-carving"
    input_path = ASSETS_DIR / "Broadway_tower_edit.jpg"
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Не смог прочесть: {str(input_path)}")

    REMOVE_COLS = 200
    REMOVE_ROWS = 30

    out = image

    out = carve_vertical(out, REMOVE_COLS)
    out = carve_horizontal(out, REMOVE_ROWS)

    outp = ASSETS_DIR / "result.jpg"
    ok = cv2.imwrite(str(outp), out)
    if not ok:
        raise RuntimeError(f"Не получилось записать: {outp}")


if __name__ == "__main__":
    main()
