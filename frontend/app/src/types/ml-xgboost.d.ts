// Local typings for ml-xgboost used by our wrapper (xgb.ts).
// This file is a proper module so that `import type { XGBoostInterface } from "./types/ml-xgboost";` works.

export interface XGBoostInterface {
  train(X: number[][], y: number[]): void | Promise<void>;
  predict(rows: number[][]): number[] | Promise<number[]>;
}

export declare class XGBoost implements XGBoostInterface {
  constructor(config?: any);
  train(X: number[][], y: number[]): void | Promise<void>;
  predict(rows: number[][]): number[] | Promise<number[]>;
}

declare const _default: {
  XGBoost: typeof XGBoost;
};

export default _default;

// Also provide typings for the actual "ml-xgboost" package name
// so that dynamic import("ml-xgboost") and mocks are type-safe.
declare module "ml-xgboost" {
  export interface XGBoostInterface {
    train(X: number[][], y: number[]): void | Promise<void>;
    predict(rows: number[][]): number[] | Promise<number[]>;
  }

  export class XGBoost implements XGBoostInterface {
    constructor(config?: any);
    train(X: number[][], y: number[]): void | Promise<void>;
    predict(rows: number[][]): number[] | Promise<number[]>;
  }

  const _default: {
    XGBoost: typeof XGBoost;
  };

  export default _default;
}
