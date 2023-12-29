using Drumwise.Application.Common.Errors;
using Drumwise.Application.Common.Models.Identity;
using FluentValidation;

namespace Drumwise.Infrastructure.Identity.Validators;

public class AdditionalUserDataRequestValidator : AbstractValidator<AdditionalUserDataRequest>
{
    public AdditionalUserDataRequestValidator()
    {
        RuleFor(x => x.Experience)
            .GreaterThanOrEqualTo(0)
            .WithMessage(Error.GetErrorMessage(ErrorCodes.Identity.ExperienceBelowZero))
            .WithErrorCode(ErrorCodes.Identity.ExperienceBelowZero);
    }
}