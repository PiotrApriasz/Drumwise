using Drumwise.Application.Common.Errors;
using FluentValidation;

namespace Drumwise.Features.Homeworks.Validation;

public class CreateHomeworkCommandValidator : AbstractValidator<CreateHomeworkCommand>
{
    public CreateHomeworkCommandValidator()
    {
        RuleFor(x => x.Title)
            .NotEmpty()
            .WithMessage(Error.GetErrorMessage(ErrorCodes.Homework.TitleIsRequired))
            .WithErrorCode(ErrorCodes.Homework.TitleIsRequired);

        RuleFor(x => x.Deadline)
            .Must(BeMoreThanOneDayLater)
            .WithMessage(Error.GetErrorMessage(ErrorCodes.Homework.ToLittleDeadline))
            .WithErrorCode(ErrorCodes.Homework.ToLittleDeadline);

        RuleFor(x => x.AssignedTo)
            .NotEmpty()
            .WithErrorCode(Error.GetErrorMessage(ErrorCodes.Homework.TitleIsRequired))
            .WithMessage(ErrorCodes.Homework.AssignedToIsRequired);

    }

    private static bool BeMoreThanOneDayLater(DateTime deadLine)
    {
        var today = DateTime.Now;
        return deadLine > today.AddDays(1);
    }
}