import {
  loadFromCSV,
  loadFromXLSX,
  trainModel,
  predictNext,
  LoadedData,
} from "../core";
import { parseXLSX, buildFeatures } from "../api";
import { initXGBoostCtor } from "../xgb";

jest.mock("../api");
jest.mock("../xgb");

// Helper to get typed mocks
const mockedParseXLSX = parseXLSX as jest.MockedFunction<typeof parseXLSX>;
const mockedBuildFeatures = buildFeatures as jest.MockedFunction<
  typeof buildFeatures
>;
const mockedInitXGBoostCtor =
  initXGBoostCtor as jest.MockedFunction<typeof initXGBoostCtor>;

describe("loadFromCSV", () => {
  it("infers datetimeKey from headers containing date/time", async () => {
    const csv = `
Date,value
2025-01-01,1
`.trim();

    const data = await loadFromCSV(csv);

    expect(data.headers).toEqual(["Date", "value"]);
    expect(data.datetimeKey).toBe("Date");
    expect(data.rows).toHaveLength(1);
  });

  it("returns null datetimeKey when no date/time-like column", async () => {
    const csv = `
x,y
1,2
`.trim();

    const data = await loadFromCSV(csv);

    expect(data.headers).toEqual(["x", "y"]);
    expect(data.datetimeKey).toBeNull();
  });
});

describe("loadFromXLSX", () => {
  it("uses parseXLSX and infers datetimeKey", async () => {
    mockedParseXLSX.mockResolvedValueOnce({
      headers: ["timestamp", "v"],
      rows: [{ timestamp: "2025-01-01", v: 1 }],
    });

    const buf = new ArrayBuffer(4);
    const data = await loadFromXLSX(buf);

    expect(mockedParseXLSX).toHaveBeenCalledTimes(1);
    expect(data.headers).toEqual(["timestamp", "v"]);
    expect(data.datetimeKey).toBe("timestamp");
    expect(data.rows).toHaveLength(1);
  });
});

describe("trainModel & predictNext", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("calls initXGBoostCtor, trains model, and predicts using lastFeatureRow", async () => {
    const rows = [
      { datetime: "t0", a: 1, target: 10 },
      { datetime: "t1", a: 2, target: 20 },
      { datetime: "t2", a: 3, target: 30 },
    ];

    const fakeFeatures = {
      X: [
        [1, 0],
        [2, 1],
        [3, 2],
      ],
      y: [10, 20, 30],
      lastFeatureRow: [3, 3],
    };

    mockedBuildFeatures.mockReturnValue(fakeFeatures);

    class FakeXGBoost {
      public trainedX: number[][] | null = null;
      public trainedY: number[] | null = null;
      constructor(public config: any) {}
      train(X: number[][], y: number[]): void {
        this.trainedX = X;
        this.trainedY = y;
      }
      predict(rowsIn: number[][]): number[] {
        // deterministic: mean of y
        const mean =
          (this.trainedY ?? []).reduce((s, v) => s + v, 0) /
          (this.trainedY ?? []).length;
        return rowsIn.map(() => mean);
      }
    }

    mockedInitXGBoostCtor.mockResolvedValueOnce(FakeXGBoost as any);

    const data: LoadedData = {
      rows,
      headers: ["datetime", "a", "target"],
      datetimeKey: "datetime",
    };

    const booster = await trainModel(data, "target");
    expect(mockedInitXGBoostCtor).toHaveBeenCalledTimes(1);
    expect(mockedBuildFeatures).toHaveBeenCalledWith(
      data.rows,
      "datetime",
      "target"
    );
    expect(booster).toBeInstanceOf(FakeXGBoost);

    const yhat = predictNext(data, "target", booster);
    // With FakeXGBoost, prediction == mean(y) = 20
    expect(yhat).toBeCloseTo(20);
  });
});
