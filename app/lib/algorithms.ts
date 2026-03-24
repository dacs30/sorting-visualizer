export type SortStep = {
  array: number[];
  ids: number[];
  comparing: number[];
  swapping: number[];
  sorted: number[];
  pivot?: number;
};

export type AlgorithmInfo = {
  name: string;
  description: string;
  howItWorks: string[];
  timeComplexity: { best: string; average: string; worst: string };
  spaceComplexity: string;
  stable: boolean;
};

export const algorithmInfo: Record<string, AlgorithmInfo> = {
  bubble: {
    name: "Bubble Sort",
    description:
      "Bubble Sort repeatedly steps through the list, compares adjacent elements, and swaps them if they're in the wrong order. The largest unsorted element 'bubbles up' to its correct position after each pass.",
    howItWorks: [
      "Start at the beginning of the array.",
      "Compare each pair of adjacent elements.",
      "If the left element is greater than the right, swap them.",
      "Continue to the end — the largest element is now at the end.",
      "Repeat for the remaining unsorted portion.",
      "Stop when no swaps occur in a full pass.",
    ],
    timeComplexity: { best: "O(n)", average: "O(n²)", worst: "O(n²)" },
    spaceComplexity: "O(1)",
    stable: true,
  },
  selection: {
    name: "Selection Sort",
    description:
      "Selection Sort divides the array into a sorted and unsorted region. It repeatedly finds the minimum element from the unsorted region and places it at the beginning of the sorted region.",
    howItWorks: [
      "Find the minimum element in the unsorted portion.",
      "Swap it with the first element of the unsorted portion.",
      "Move the boundary of the sorted region one step to the right.",
      "Repeat until the entire array is sorted.",
    ],
    timeComplexity: { best: "O(n²)", average: "O(n²)", worst: "O(n²)" },
    spaceComplexity: "O(1)",
    stable: false,
  },
  insertion: {
    name: "Insertion Sort",
    description:
      "Insertion Sort builds the sorted array one item at a time. It picks each element and inserts it into its correct position within the already-sorted portion, shifting larger elements to make room.",
    howItWorks: [
      "Start with the second element (the first is trivially sorted).",
      "Pick the current element as the 'key'.",
      "Compare the key with elements in the sorted portion (moving right to left).",
      "Shift all elements greater than the key one position to the right.",
      "Insert the key in the correct position.",
      "Move to the next element and repeat.",
    ],
    timeComplexity: { best: "O(n)", average: "O(n²)", worst: "O(n²)" },
    spaceComplexity: "O(1)",
    stable: true,
  },
  merge: {
    name: "Merge Sort",
    description:
      "Merge Sort is a divide-and-conquer algorithm. It splits the array in half recursively until individual elements remain, then merges them back together in sorted order.",
    howItWorks: [
      "Divide the array into two halves.",
      "Recursively sort each half.",
      "Merge the two sorted halves: compare the front elements of each half and pick the smaller one.",
      "Continue until both halves are exhausted.",
      "The merged result is a fully sorted array.",
    ],
    timeComplexity: { best: "O(n log n)", average: "O(n log n)", worst: "O(n log n)" },
    spaceComplexity: "O(n)",
    stable: true,
  },
  quick: {
    name: "Quick Sort",
    description:
      "Quick Sort picks a 'pivot' element and partitions the array so all elements smaller than the pivot come before it and all larger elements come after. It then recursively sorts the two partitions.",
    howItWorks: [
      "Choose a pivot element (here: last element).",
      "Partition: move all elements smaller than the pivot to its left, larger to its right.",
      "Place the pivot in its final sorted position.",
      "Recursively apply the same process to the left and right sub-arrays.",
      "Base case: a sub-array of size 0 or 1 is already sorted.",
    ],
    timeComplexity: { best: "O(n log n)", average: "O(n log n)", worst: "O(n²)" },
    spaceComplexity: "O(log n)",
    stable: false,
  },
  heap: {
    name: "Heap Sort",
    description:
      "Heap Sort uses a binary heap data structure. It first builds a max-heap from the array, then repeatedly extracts the maximum element and places it at the end, shrinking the heap each time.",
    howItWorks: [
      "Build a max-heap from the array (parent ≥ children).",
      "The root of the heap is the largest element.",
      "Swap the root with the last element of the heap.",
      "Reduce the heap size by 1 and 'heapify' the root to restore the heap property.",
      "Repeat until the heap has one element.",
    ],
    timeComplexity: { best: "O(n log n)", average: "O(n log n)", worst: "O(n log n)" },
    spaceComplexity: "O(1)",
    stable: false,
  },
};

function addStep(
  steps: SortStep[],
  array: number[],
  ids: number[],
  comparing: number[],
  swapping: number[],
  sorted: number[],
  pivot?: number
) {
  steps.push({ array: [...array], ids: [...ids], comparing, swapping, sorted: [...sorted], pivot });
}

export function bubbleSortSteps(input: number[]): SortStep[] {
  const arr = [...input];
  const ids = arr.map((_, i) => i);
  const steps: SortStep[] = [];
  const sorted: number[] = [];
  const n = arr.length;

  for (let i = 0; i < n - 1; i++) {
    let swapped = false;
    for (let j = 0; j < n - 1 - i; j++) {
      addStep(steps, arr, ids, [j, j + 1], [], sorted);
      if (arr[j] > arr[j + 1]) {
        [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];
        [ids[j], ids[j + 1]] = [ids[j + 1], ids[j]];
        addStep(steps, arr, ids, [], [j, j + 1], sorted);
        swapped = true;
      }
    }
    sorted.unshift(n - 1 - i);
    addStep(steps, arr, ids, [], [], sorted);
    if (!swapped) break;
  }
  if (!sorted.includes(0)) sorted.unshift(0);
  addStep(steps, arr, ids, [], [], Array.from({ length: n }, (_, i) => i));
  return steps;
}

export function selectionSortSteps(input: number[]): SortStep[] {
  const arr = [...input];
  const ids = arr.map((_, i) => i);
  const steps: SortStep[] = [];
  const sorted: number[] = [];
  const n = arr.length;

  for (let i = 0; i < n - 1; i++) {
    let minIdx = i;
    for (let j = i + 1; j < n; j++) {
      addStep(steps, arr, ids, [minIdx, j], [], sorted);
      if (arr[j] < arr[minIdx]) minIdx = j;
    }
    if (minIdx !== i) {
      [arr[i], arr[minIdx]] = [arr[minIdx], arr[i]];
      [ids[i], ids[minIdx]] = [ids[minIdx], ids[i]];
      addStep(steps, arr, ids, [], [i, minIdx], sorted);
    }
    sorted.push(i);
    addStep(steps, arr, ids, [], [], sorted);
  }
  addStep(steps, arr, ids, [], [], Array.from({ length: n }, (_, i) => i));
  return steps;
}

export function insertionSortSteps(input: number[]): SortStep[] {
  const arr = [...input];
  const ids = arr.map((_, i) => i);
  const steps: SortStep[] = [];
  const sorted: number[] = [0];
  const n = arr.length;

  for (let i = 1; i < n; i++) {
    const key = arr[i];
    const keyId = ids[i];
    let j = i - 1;
    addStep(steps, arr, ids, [i], [], sorted);
    while (j >= 0 && arr[j] > key) {
      addStep(steps, arr, ids, [j, j + 1], [], sorted);
      arr[j + 1] = arr[j];
      ids[j + 1] = ids[j];
      addStep(steps, arr, ids, [], [j, j + 1], sorted);
      j--;
    }
    arr[j + 1] = key;
    ids[j + 1] = keyId;
    sorted.push(i);
    addStep(steps, arr, ids, [], [], sorted);
  }
  addStep(steps, arr, ids, [], [], Array.from({ length: n }, (_, i) => i));
  return steps;
}

export function mergeSortSteps(input: number[]): SortStep[] {
  const arr = [...input];
  const ids = arr.map((_, i) => i);
  const steps: SortStep[] = [];
  const sorted: number[] = [];
  const n = arr.length;

  function merge(left: number, mid: number, right: number) {
    const leftArr = arr.slice(left, mid + 1);
    const rightArr = arr.slice(mid + 1, right + 1);
    const leftIds = ids.slice(left, mid + 1);
    const rightIds = ids.slice(mid + 1, right + 1);
    let i = 0, j = 0, k = left;

    while (i < leftArr.length && j < rightArr.length) {
      addStep(steps, arr, ids, [left + i, mid + 1 + j], [], sorted);
      if (leftArr[i] <= rightArr[j]) {
        arr[k] = leftArr[i];
        ids[k] = leftIds[i];
        i++;
      } else {
        arr[k] = rightArr[j];
        ids[k] = rightIds[j];
        j++;
      }
      addStep(steps, arr, ids, [], [k], sorted);
      k++;
    }
    while (i < leftArr.length) {
      arr[k] = leftArr[i];
      ids[k] = leftIds[i];
      k++; i++;
      addStep(steps, arr, ids, [], [k - 1], sorted);
    }
    while (j < rightArr.length) {
      arr[k] = rightArr[j];
      ids[k] = rightIds[j];
      k++; j++;
      addStep(steps, arr, ids, [], [k - 1], sorted);
    }
  }

  function mergeSort(left: number, right: number) {
    if (left >= right) return;
    const mid = Math.floor((left + right) / 2);
    mergeSort(left, mid);
    mergeSort(mid + 1, right);
    merge(left, mid, right);
    for (let i = left; i <= right; i++) sorted.push(i);
  }

  mergeSort(0, n - 1);
  addStep(steps, arr, ids, [], [], Array.from({ length: n }, (_, i) => i));
  return steps;
}

export function quickSortSteps(input: number[]): SortStep[] {
  const arr = [...input];
  const ids = arr.map((_, i) => i);
  const steps: SortStep[] = [];
  const sorted: number[] = [];
  const n = arr.length;

  function partition(low: number, high: number): number {
    const pivot = arr[high];
    let i = low - 1;

    for (let j = low; j < high; j++) {
      addStep(steps, arr, ids, [j, high], [], sorted, high);
      if (arr[j] < pivot) {
        i++;
        [arr[i], arr[j]] = [arr[j], arr[i]];
        [ids[i], ids[j]] = [ids[j], ids[i]];
        addStep(steps, arr, ids, [], [i, j], sorted, high);
      }
    }
    [arr[i + 1], arr[high]] = [arr[high], arr[i + 1]];
    [ids[i + 1], ids[high]] = [ids[high], ids[i + 1]];
    addStep(steps, arr, ids, [], [i + 1, high], sorted, i + 1);
    sorted.push(i + 1);
    return i + 1;
  }

  function quickSort(low: number, high: number) {
    if (low < high) {
      const pi = partition(low, high);
      quickSort(low, pi - 1);
      quickSort(pi + 1, high);
    } else if (low === high) {
      sorted.push(low);
    }
  }

  quickSort(0, n - 1);
  addStep(steps, arr, ids, [], [], Array.from({ length: n }, (_, i) => i));
  return steps;
}

export function heapSortSteps(input: number[]): SortStep[] {
  const arr = [...input];
  const ids = arr.map((_, i) => i);
  const steps: SortStep[] = [];
  const sorted: number[] = [];
  const n = arr.length;

  function heapify(size: number, root: number) {
    let largest = root;
    const left = 2 * root + 1;
    const right = 2 * root + 2;

    addStep(steps, arr, ids, [root, left < size ? left : root], [], sorted);
    if (left < size && arr[left] > arr[largest]) largest = left;
    if (right < size && arr[right] > arr[largest]) largest = right;

    if (largest !== root) {
      [arr[root], arr[largest]] = [arr[largest], arr[root]];
      [ids[root], ids[largest]] = [ids[largest], ids[root]];
      addStep(steps, arr, ids, [], [root, largest], sorted);
      heapify(size, largest);
    }
  }

  for (let i = Math.floor(n / 2) - 1; i >= 0; i--) heapify(n, i);

  for (let i = n - 1; i > 0; i--) {
    [arr[0], arr[i]] = [arr[i], arr[0]];
    [ids[0], ids[i]] = [ids[i], ids[0]];
    sorted.push(i);
    addStep(steps, arr, ids, [], [0, i], sorted);
    heapify(i, 0);
  }
  sorted.push(0);
  addStep(steps, arr, ids, [], [], Array.from({ length: n }, (_, i) => i));
  return steps;
}

export const algorithmFunctions: Record<string, (arr: number[]) => SortStep[]> = {
  bubble: bubbleSortSteps,
  selection: selectionSortSteps,
  insertion: insertionSortSteps,
  merge: mergeSortSteps,
  quick: quickSortSteps,
  heap: heapSortSteps,
};
