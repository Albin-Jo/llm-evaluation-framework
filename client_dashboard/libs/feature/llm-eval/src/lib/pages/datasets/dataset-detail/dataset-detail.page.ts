/* Path: libs/feature/llm-eval/src/lib/pages/datasets/dataset-detail/dataset-detail.page.scss */
@import '../../../../../../../styles/variables';
@import '../../../../../../../styles/mixins';

:host {
  display: block;
  width: 100%;
}

.dataset-detail-container {
  padding: $spacing-6;
  max-width: $container-xxl;
  margin: 0 auto;

  @include media-breakpoint-down(md) {
    padding: $spacing-4;
  }
}

// Back Navigation
.back-navigation {
  margin-bottom: $spacing-6;

  .back-button {
    display: inline-flex;
    align-items: center;
    background: none;
    border: none;
    color: $primary;
    font-size: $font-size-sm;
    padding: $spacing-2 0;
    cursor: pointer;

    &:hover {
      text-decoration: underline;
    }

    .back-icon {
      margin-right: $spacing-2;
      font-size: $font-size-md;
    }
  }

  // Document Preview Modal
  .document-preview-modal {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 1000;
    display: flex;
    align-items: center;
    justify-content: center;

    .modal-backdrop {
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background-color: rgba(0, 0, 0, 0.5);
    }

    .modal-content {
      position: relative;
      width: 90%;
      max-width: 800px;
      max-height: 90vh;
      background-color: white;
      border-radius: $radius-md;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      box-shadow: $shadow-lg;

      .modal-header {
        padding: $spacing-4 $spacing-6;
        border-bottom: 1px solid $border-light;
        display: flex;
        justify-content: space-between;
        align-items: center;

        .modal-title {
          margin: 0;
          font-size: $font-size-lg;
          font-weight: $font-weight-semibold;
        }

        .close-button {
          background: none;
          border: none;
          font-size: $font-size-xl;
          cursor: pointer;
          color: $text-secondary;

          &:hover {
            color: $text-primary;
          }
        }
      }

      .modal-body {
        flex: 1;
        overflow: auto;
        padding: $spacing-4 $spacing-6;

        .preview-content {
          height: 100%;
          min-height: 300px;

          .preview-loading, .preview-error {
            height: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: $text-secondary;

            .spinner {
              @include spinner;
              margin-bottom: $spacing-4;
            }

            .error-icon {
              font-size: $font-size-2xl;
              color: $error;
              margin-bottom: $spacing-4;
            }
          }

          .preview-container {
            height: 100%;

            .csv-preview, .text-preview, .json-preview, .fallback-preview {
              height: 100%;
              overflow: auto;
            }

            .csv-table {
              border: 1px solid $border-light;
              border-radius: $radius-sm;
              overflow: hidden;

              .csv-header {
                display: flex;
                background-color: $background-light;
                font-weight: $font-weight-semibold;

                .csv-cell {
                  padding: $spacing-3;
                  flex: 1;
                  min-width: 100px;
                  border-right: 1px solid $border-light;

                  &:last-child {
                    border-right: none;
                  }
                }
              }

              .csv-row {
                display: flex;
                border-top: 1px solid $border-light;

                &:nth-child(even) {
                  background-color: $background-light;
                }

                .csv-cell {
                  padding: $spacing-3;
                  flex: 1;
                  min-width: 100px;
                  border-right: 1px solid $border-light;
                  word-break: break-word;

                  &:last-child {
                    border-right: none;
                  }
                }
              }
            }

            .text-preview, .json-preview {
              pre {
                margin: 0;
                padding: $spacing-4;
                background-color: $background-light;
                border-radius: $radius-sm;
                font-family: monospace;
                white-space: pre-wrap;
                word-break: break-word;
              }
            }

            .fallback-preview {
              display: flex;
              flex-direction: column;
              align-items: center;
              justify-content: center;
              text-align: center;
              color: $text-secondary;
              padding: $spacing-8;

              p:first-child {
                margin-bottom: $spacing-3;
              }
            }
          }
        }
      }

      .modal-footer {
        padding: $spacing-4 $spacing-6;
        border-top: 1px solid $border-light;
        display: flex;
        justify-content: flex-end;
        gap: $spacing-3;

        .outline-button {
          @include button-outline;
        }

        .primary-button {
          @include button-primary;
        }
      }
    }
  }
}

// Loading & Error States
.loading-container, .error-container {
  @include card-padded;
  margin-bottom: $spacing-6;
  text-align: center;
  padding: $spacing-12;
}

.loading-container {
  @include loading-container;
}

.error-container {
  @include error-container;

  h2 {
    color: $error;
    margin-bottom: $spacing-2;
  }

  p {
    margin-bottom: $spacing-6;
  }

  .error-actions {
    display: flex;
    justify-content: center;
    gap: $spacing-4;
  }

  .primary-button {
    @include button-primary;
  }
}

// Dataset Details
.dataset-details {
  display: flex;
  flex-direction: column;
  gap: $spacing-6;

  // Header with Title and Actions
  .detail-header {
    @include card-padded;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: $spacing-4 $spacing-6;

    @include media-breakpoint-down(md) {
      flex-direction: column;
      align-items: flex-start;
      gap: $spacing-4;
    }

    .header-content {
      flex: 1;

      .detail-title {
        margin: 0 0 $spacing-2 0;
        font-size: $font-size-2xl;
        font-weight: $font-weight-semibold;
        color: $text-primary;
      }

      .detail-title-input {
        width: 100%;
        max-width: 500px;
        font-size: $font-size-2xl;
        font-weight: $font-weight-semibold;
        color: $text-primary;
        padding: $spacing-2;
        border: 1px solid $border-color;
        border-radius: $radius-sm;
        margin-bottom: $spacing-2;

        &:focus {
          outline: none;
          border-color: $primary;
        }

        &.is-invalid {
          border-color: $error;
        }
      }

      .detail-badges {
        display: flex;
        flex-wrap: wrap;
        gap: $spacing-2;
      }

      .status-badge {
        padding: $spacing-1 $spacing-3;
        border-radius: $radius-full;
        font-size: $font-size-xs;
        font-weight: $font-weight-medium;
        text-transform: uppercase;

        &.status-ready {
          background-color: $success-light;
          color: $success;
        }

        &.status-processing {
          background-color: $warning-light;
          color: darken($warning, 10%);
        }

        &.status-error {
          background-color: $error-light;
          color: $error;
        }
      }
    }

    .header-actions {
      display: flex;
      gap: $spacing-3;

      @include media-breakpoint-down(md) {
        width: 100%;
      }

      .outline-button {
        @include button-outline;

        &.edit-button {
          color: $primary;
          border-color: $primary;
        }

        &.delete-button {
          color: $error;
          border-color: $error;
        }

        &.cancel-button {
          color: $text-secondary;
          border-color: $border-color;
        }
      }

      .primary-button {
        @include button-primary;
      }
    }
  }

  // Main Content Layout
  .content-layout {
    display: flex;
    gap: $spacing-6;

    @include media-breakpoint-down(lg) {
      flex-direction: column;
    }

    .left-column, .right-column {
      display: flex;
      flex-direction: column;
      gap: $spacing-6;
    }

    .left-column {
      flex: 3;
    }

    .right-column {
      flex: 2;
    }
  }

  // Content Cards
  .content-card {
    background-color: white;
    border: 1px solid $border-color;
    border-radius: $radius-sm;
    padding: $spacing-6;
    box-shadow: $shadow-sm;

    @include media-breakpoint-down(md) {
      padding: $spacing-4;
    }

    .card-title {
      font-size: $font-size-lg;
      font-weight: $font-weight-semibold;
      color: $text-primary;
      margin-top: 0;
      margin-bottom: $spacing-4;
      padding-bottom: $spacing-3;
      border-bottom: 1px solid $border-light;
    }
  }

  // Dataset Info Card
  .dataset-info-card {
    .info-group {
      margin-bottom: $spacing-4;

      &:last-child {
        margin-bottom: 0;
      }

      .info-label {
        font-size: $font-size-sm;
        color: $text-secondary;
        margin-bottom: $spacing-1;
      }

      .info-value {
        color: $text-primary;
      }

      .info-textarea {
        width: 100%;
        min-height: 80px;
        padding: $spacing-3;
        border: 1px solid $border-color;
        border-radius: $radius-sm;
        font-size: $font-size-sm;
        resize: vertical;

        &:focus {
          outline: none;
          border-color: $primary;
        }
      }
    }

    .info-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: $spacing-4;
      margin-bottom: $spacing-4;

      .info-item {
        .info-label {
          font-size: $font-size-sm;
          color: $text-secondary;
          margin-bottom: $spacing-1;
        }

        .info-value {
          color: $text-primary;
        }
      }
    }

    // Tags Section
    .tags-container {
      display: flex;
      flex-wrap: wrap;
      gap: $spacing-2;

      .tag-pill {
        display: inline-flex;
        padding: $spacing-1 $spacing-3;
        border-radius: $radius-full;
        background-color: $primary-light;
        color: $primary;
        font-size: $font-size-sm;
      }

      .no-tags {
        color: $text-secondary;
        font-style: italic;
        font-size: $font-size-sm;
      }

      .tag-button {
        background-color: $background-light;
        border: 1px solid $border-color;
        border-radius: $radius-full;
        padding: $spacing-2 $spacing-3;
        font-size: $font-size-sm;
        color: $text-secondary;
        cursor: pointer;
        transition: all 0.2s;

        &:hover {
          background-color: darken($background-light, 5%);
        }

        &.active {
          background-color: $primary-light;
          border-color: $primary;
          color: $primary;
        }
      }
    }
  }

  // Statistics Card
  .statistics-card {
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: $spacing-4;

      @include media-breakpoint-down(sm) {
        grid-template-columns: 1fr;
      }

      .stat-item {
        .stat-card {
          background-color: $background-light;
          border-radius: $radius-sm;
          padding: $spacing-4;

          .stat-label {
            font-size: $font-size-sm;
            color: $text-secondary;
            margin-bottom: $spacing-2;
          }

          .stat-value {
            font-size: $font-size-2xl;
            font-weight: $font-weight-bold;
            color: $primary;

            .stat-unit {
              font-size: $font-size-sm;
              color: $text-secondary;
              font-weight: $font-weight-regular;
              margin-left: $spacing-1;
            }
          }

          .stat-value-small {
            font-size: $font-size-base;
            color: $text-primary;
          }
        }
      }
    }
  }

  // Documents Card
  .documents-card {
    .empty-documents {
      @include empty-state;
      padding: $spacing-8;
      text-align: center;

      .empty-icon {
        font-size: $font-size-3xl;
        color: $text-tertiary;
        margin-bottom: $spacing-4;
      }

      p {
        color: $text-secondary;
        margin-bottom: $spacing-6;
      }

      .primary-button {
        @include button-primary;
      }
    }

    .document-display {
      .document-list {
        width: 100%;
        overflow-x: auto;
      }

      .table-header {
        display: flex;
        background-color: $background-light;
        border: 1px solid $border-color;
        font-weight: $font-weight-semibold;
        color: $text-secondary;
        font-size: $font-size-xs;
        text-transform: uppercase;
        letter-spacing: 0.5px;

        > div {
          padding: $spacing-3 $spacing-4;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
      }

      .table-body {
        .table-row {
          display: flex;
          border-bottom: 1px solid $border-light;
          align-items: center;
          transition: background-color 0.2s;

          &:hover {
            background-color: rgba($primary, 0.05);
          }

          > div {
            padding: $spacing-3 $spacing-4;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
          }
        }
      }

      // Column widths
      .filename-col {
        width: 35%;
        min-width: 180px;
      }

      .format-col {
        width: 10%;
        min-width: 80px;
      }

      .size-col {
        width: 15%;
        min-width: 80px;
      }

      .added-col {
        width: 20%;
        min-width: 120px;
      }

      .actions-col {
        width: 20%;
        min-width: 160px;
        display: flex;
        justify-content: flex-end;
        gap: $spacing-2;

        .action-button {
          padding: $spacing-1 $spacing-3;
          border: 1px solid $border-color;
          border-radius: $radius-sm;
          background: none;
          font-size: $font-size-xs;
          cursor: pointer;
          transition: $transition-normal;

          &:hover {
            background-color: $background-light;
          }

          &.view-button:hover {
            border-color: $primary;
            color: $primary;
          }

          &.delete-button:hover {
            border-color: $error;
            color: $error;
          }
        }
      }
    }
  }
}
