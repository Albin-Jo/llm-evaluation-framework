import { ComponentFixture, TestBed } from '@angular/core/testing';
import { LlmEvalComponent } from './llm-eval.component';

describe('LlmEvalComponent', () => {
  let component: LlmEvalComponent;
  let fixture: ComponentFixture<LlmEvalComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [LlmEvalComponent],
    }).compileComponents();

    fixture = TestBed.createComponent(LlmEvalComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
