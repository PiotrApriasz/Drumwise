using FluentValidation;

namespace Drumwise.Features.Homeworks.Validation;

public class GetHomeworkQueryValidator : AbstractValidator<GetHomeworkQuery>
{
    public GetHomeworkQueryValidator()
    {
        RuleFor(x => x.HomeworkId)
            .NotEmpty()
            .WithMessage("Homework Id must be provided")
            .WithErrorCode("Homework.HomeworkIdRequired")
            .Must(BeValidaGuid)
            .WithMessage("Homework Id is in invalid format")
            .WithErrorCode("Common.InvalidFormat");
    }

    private bool BeValidaGuid(string guidString)
    {
        return Guid.TryParse(guidString, out _);
    }
}