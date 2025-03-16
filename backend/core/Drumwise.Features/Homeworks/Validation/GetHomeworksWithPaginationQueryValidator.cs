using FluentValidation;

namespace Drumwise.Features.Homeworks.Validation;

public class GetHomeworksWithPaginationQueryValidator : AbstractValidator<GetHomeworksWithPaginationQuery>
{
    public GetHomeworksWithPaginationQueryValidator()
    {
        RuleFor(x => x.PageNumber)
            .GreaterThanOrEqualTo(1)
            .WithMessage("Page Number must be greater or equal to 1")
            .WithErrorCode("Homework.IncorrectPageNumber");

        RuleFor(x => x.PageSize)
            .GreaterThanOrEqualTo(1)
            .WithMessage("Page Size must be greater or equal to 1")
            .WithErrorCode("Homework.IncorrectPageSize");
    }
}